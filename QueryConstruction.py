import datetime
from typing import Literal, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI



load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'RAG_Modular_Script'
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')  

class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )

    #printing only the fields that are not None and not equal to their default value (if they have one)
    def pretty_print(self) -> None:
        for field in type(self).model_fields:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                type(self).model_fields[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")


def construct_query(llm):
    system = """You are an expert at converting user questions into database queries. \
    You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
    Given a question, return a database query optimized to retrieve the most relevant results.

    If there are acronyms or words you are not familiar with, do not try to rephrase them."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    structured_llm = llm.with_structured_output(TutorialSearch)
    query_analyzer = prompt | structured_llm
    return query_analyzer

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o.mini", temperature=0)
    query_analyzer = construct_query(llm)

    question = "What are some good tutorials for learning how to use LangSmith?"
    query = query_analyzer(question=question)
    query.pretty_print()


