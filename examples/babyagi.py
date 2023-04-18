"""This example is a simplified translation of https://github.com/yoheinakajima/babyagi"""
from collections import deque
from dataclasses import dataclass
from typing import Callable, List

import outlines.text as text

MODEL = "openai/gpt-3.5-turbo"


@dataclass
class LLMFunction:
    model_name: str
    prompt_fn: Callable
    format_fn: Callable = lambda x: x

    def __call__(self, *args, **kwargs):
        prompt = self.prompt_fn(*args, **kwargs)
        model, init_state = text.chat_completion(self.model_name)
        result, _ = model(prompt, init_state)
        return self.format_fn(result)


#################
# Perform tasks #
#################


@text.prompt
def perform_task_ppt(objective: str, task: str):
    """You are an AI who performs one task based on the following objective: {{objective}}.

    Your task: {{task.task_name}}

    Response:
    """


perform_task = LLMFunction(MODEL, perform_task_ppt)


#####################
# Create a new task #
#####################


@text.prompt
def create_tasks_ppt(
    objective: str, previous_task: str, result: str, task_list: List[str]
):
    """You are an task creation AI that uses the result of an execution agent to \
    create new tasks with the following objective: {{objective}}.

    The last completed task has the result: {{result}}. This result was based on this task \
    description: {{previous_task}}.

    These are incomplete tasks: {{task_list | join(task_list)}}.

    Based on the result, create new tasks to be completed by the AI system that \
    do not overlap with incomplete tasks. Return the tasks as an array.
    """


def create_tasks_fmt(result):
    new_tasks = result.split("\n")

    task_list = []
    for task in new_tasks:
        parts = task.strip().split(".", 1)
        if len(parts) == 2:
            task_list.append(parts[1].strip())

    return task_list


create_tasks = LLMFunction(MODEL, create_tasks_ppt, create_tasks_fmt)


########################
# Prioritize new tasks #
########################


@text.prompt
def prioritize_tasks_ppt(objective: str, task_names: List[str], next_task_id: int):
    """You are an task prioritization AI tasked with cleaning the formatting of \
    and reprioritizing the following tasks: {{task_names}}. Consider the ultimate \
    objective of your team: {{objective}}. Do not remove any tasks. Return the \
    result as a numbered list starting at {{next_task_id}}, like:
    #. First task
    #. Second task
    """


def prioritize_tasks_fmt(result):
    new_tasks = result.split("\n")

    task_list = deque([])
    for task in new_tasks:
        parts = task.strip().split(".", 1)
        if len(parts) == 2:
            task_id = parts[0].strip()
            task_name = parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})

    return task_list


prioritize_tasks = LLMFunction(MODEL, prioritize_tasks_ppt, prioritize_tasks_fmt)


task_id_counter = 1
objective = "Becoming rich while doing nothing."
first_task = {
    "task_id": 1,
    "task_name": "Find a repeatable, low-maintainance, scalable business.",
}
task_list = deque([first_task])


def one_cycle(objective, task_list, task_id_counter):
    """One BabyAGI cycle.

    It consists in executing the highest-priority task, creating some new tasks
    given the result, and re-priotizing the tasks.

    Parameters
    ----------
    objective
        The overall objective of the session.
    task_list
        The current list of tasks to perform.
    task_id_counter
        The current task id.

    """

    task = task_list.popleft()
    result = perform_task(objective, task)
    new_tasks = create_tasks(
        objective, first_task["task_name"], result, [first_task["task_name"]]
    )
    for task in new_tasks:
        task_id_counter += 1
        task_list.append({"task_id": task_id_counter, "task_name": task})

    prioritized_tasks = prioritize_tasks(
        objective, [task["task_name"] for task in task_list], task_id_counter
    )

    return task, result, prioritized_tasks


# Let's run it for 5 cycles to see how it works without spending a fortune.
for _ in range(5):
    task, result, task_list = one_cycle(objective, task_list, task_id_counter)
    print(f"-------\n\nTASK:\n\n{task}\n\nRESULT:\n\n {result}\n\n")
