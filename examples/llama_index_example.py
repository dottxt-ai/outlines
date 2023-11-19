"""
This example intends to show the use of a simple case of using llama_index with outlines
It relies on one of the examples proposed by llama_index: https://github.com/run-llama/llama_index/tree/main/examples/paul_graham_essay
"""
import outlines.text.generate as generate
import outlines.models as models
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from outlines.tools.llama_index import LlamaIndexOutlinesLLM

# llama_index setup
documents = SimpleDirectoryReader("data/paul_graham_essay").load_data()
index = VectorStoreIndex.from_documents(documents=documents)
service_context = ServiceContext.from_defaults(llm=LlamaIndexOutlinesLLM())
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.SIMPLE_SUMMARIZE,
    service_context=service_context
)
query_engine = index.as_query_engine(response_synthesizer=response_synthesizer)

model = models.transformers("gpt2", llama_index_engine=query_engine)
prompt = "What did the author do after he left YC? Choose one among the following choices: Painting, Running"
answer = generate.choice(model, ["Painting", "Running"])(prompt)
print(answer)
