"""This example is a simplified translation of BabyAGI.

It currently does not use the vector store retrieval

The original repo can be found at https://github.com/yoheinakajima/babyagi
"""
from collections import deque
from typing import Deque, List

import outlines.models as models
import outlines.text as text

model = models.text_completion.openai("gpt-3.5-turbo")


#################
# Perform tasks #
#################


@text.prompt
def perform_task_ppt(objective: str, task: str):
    """You are an AI who performs one task based on the following objective: {{objective}}.

    Your task: {{task.task_name}}

    Response:
    """


perform_task = text.function(model, perform_task_ppt)


#####################
# Create a new task #
#####################


@text.prompt
def create_tasks_ppt(
    objective: str, previous_task: str, result: str, task_list: List[str]
):
    """You are an task creation AI that uses the result of an execution agent to \
    create new tasks with the following objective: {{objective}}.

    The last completed task has the result: {{result}}.

    This result was based on this task description: {{previous_task}}. These are \
    incomplete tasks: {{task_list | join(task_list)}}.

    Based on the result, create new tasks to be completed by the AI system that \
    do not overlap with incomplete tasks.

    Return the tasks as an array.
    """


def create_tasks_fmt(result: str) -> List[str]:
    new_tasks = result.split("\n")

    task_list = []
    for task in new_tasks:
        parts = task.strip().split(".", 1)
        if len(parts) == 2:
            task_list.append(parts[1].strip())

    return task_list


create_tasks = text.function(model, create_tasks_ppt, create_tasks_fmt)


########################
# Prioritize new tasks #
########################


@text.prompt
def prioritize_tasks_ppt(objective: str, task_names: List[str], next_task_id: int):
    """You are a task prioritization AI tasked with cleaning the formatting of \
    and reprioritizing the following tasks: {{task_names}}.

    Consider the ultimate objective of your team: {{objective}}.

    Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task

    Start the tasks list with the number {{next_task_id}}.
    """


def prioritize_tasks_fmt(result: str):
    new_tasks = result.split("\n")

    task_list: Deque = deque([])
    for task in new_tasks:
        parts = task.strip().split(".", 1)
        if len(parts) == 2:
            task_id = int(parts[0].strip())
            task_name = parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})

    return task_list


prioritize_tasks = text.function(model, prioritize_tasks_ppt, prioritize_tasks_fmt)


objective = "Becoming rich while doing nothing."
first_task = {
    "task_id": 1,
    "task_name": "Find a repeatable, low-maintainance, scalable business.",
}
next_task_id = 1
task_list = deque([first_task])


def one_cycle(objective: str, task_list, next_task_id: int):
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
        next_task_id += 1
        task_list.append({"task_id": next_task_id, "task_name": task})

    prioritized_tasks = prioritize_tasks(
        objective, [task["task_name"] for task in task_list], next_task_id
    )

    return task, result, prioritized_tasks, next_task_id


# Let's run it for 5 cycles to see how it works without spending a fortune.
for _ in range(5):
    print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
    for t in task_list:
        print(" • " + str(t["task_name"]))

    task, result, task_list, next_task_id = one_cycle(
        objective, task_list, next_task_id
    )

    print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
    print(task)
    print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
    print(result)
