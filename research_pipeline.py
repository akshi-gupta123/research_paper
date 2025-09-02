import os
import langchain_core
import requests
import json
import arxiv
import PyPDF2
import io
import re
import markdown
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from xhtml2pdf import pisa

# Load environment variables
load_dotenv()

class ResearchPaperPipeline:
    def __init__(self, topic: str):
        self.topic = topic
        self.papers = []
        self.full_texts = []
        self.knowledge_base = []
        self.references = []
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_DEPLOYMENT")
        
        # Initialize the OpenAI model
        self.model = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            openai_api_version="2024-02-01",
            api_key=self.api_key
        )

    def fetch_and_parse_pdfs(self):
        """Downloads and extracts text from the found PDFs."""
        for paper in self.papers:
            try:
                response = requests.get(paper.pdf_url, timeout=30)
                response.raise_for_status()

                with io.BytesIO(response.content) as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"

                self.full_texts.append({
                    'title': paper.title,
                    'authors': [a.name for a in paper.authors],
                    'published': paper.published,
                    'summary': paper.summary,
                    'full_text': text,
                    'url': paper.pdf_url,
                    'entry_id': paper.entry_id
                })
                print(f"Fetched and parsed: {paper.title}")
            except Exception as e:
                print(f"Failed to process {paper.title}: {e}")

    def build_knowledge_base(self):
        """Chunks the full text of papers into smaller, citable sentences/paragraphs."""
        for paper_info in self.full_texts:
            text = paper_info['full_text']
            chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            for chunk in chunks:
                if len(chunk) > 50:
                    self.knowledge_base.append({
                        'text': chunk,
                        'source_title': paper_info['title'],
                        'source_authors': paper_info['authors'],
                        'source_url': paper_info['url'],
                        'source_id': paper_info['entry_id']
                    })
        print(f"Built knowledge base with {len(self.knowledge_base)} citable chunks.")

    def generate_structured_paper(self) -> str:
        """Uses the OpenAI model to generate the research paper by querying the knowledge base."""
        section_heads = ["Abstract", "Introduction", "Related Work", "Methodology", "Results", "Conclusion"]
        paper_markdown = f"# {self.topic}\n\n"

        for section in section_heads:
            relevant_chunks = self.knowledge_base[:3]  # Get relevant chunks for this section

            section_prompt = f"""
            Write the **{section}** section for a two-page research paper on **{self.topic}**.
            Use the following excerpts from research papers as your evidence:

            {self._format_chunks_for_prompt(relevant_chunks)}

            Keep the response concise, aiming for no more than 200 words for this section:
            """

            section_text = self._query_llm(section_prompt)
            paper_markdown += f"## {section}\n\n{section_text}\n\n"

            # Store the references used in this section
            self.references.extend(relevant_chunks)

        return paper_markdown

    def _format_chunks_for_prompt(self, chunks):
        """Formats knowledge base chunks for the model prompt."""
        formatted_text = ""
        for idx, chunk in enumerate(chunks):
            formatted_text += f"[Source ID: {idx + 1}]: {chunk['text']}\n\n"
        return formatted_text

    def _query_llm(self, prompt: str) -> str:
        """Helper function to query the OpenAI API and return the content section without headings."""
        messages = [
            SystemMessage(content="You are a helpful assistant that writes research papers."),
            HumanMessage(content=prompt)
        ]
        try:
            result = self.model.invoke(messages)

            # Check if the result is an instance of AIMessage
            if isinstance(result, langchain_core.messages.ai.AIMessage):
                content = result.content
                
                content = re.sub(r'\*\*.*?\*\*\n+', '', content)
                content = content.replace("Related Work", "")
                content = content.replace("Methodology", "")

                return content
            else:
                return 'Unexpected result format'
                    
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return "This is a placeholder text for the research paper section."

    def output_pdf(self, markdown_text: str, filename: str = "research_paper.pdf"):
        """Converts the generated markdown to a PDF file using xhtml2pdf."""
        # Convert markdown to HTML
        html_text = markdown.markdown(markdown_text)

        # Add basic CSS styling for better PDF formatting
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                    font-size: 12pt;
                }}
                h1 {{
                    font-size: 18pt;
                    text-align: center;
                    margin-bottom: 30px;
                    color: #2c3e50;
                }}
                h2 {{
                    font-size: 16pt;
                    margin-top: 25px;
                    margin-bottom: 15px;
                    color: #34495e;
                    border-bottom: 1px solid #ecf0f1;
                    padding-bottom: 5px;
                }}
                p {{
                    margin-bottom: 15px;
                    text-align: justify;
                }}
                .references {{
                    margin-top: 30px;
                    font-size: 11pt;
                }}
                .references p {{
                    margin-bottom: 8px;
                }}
            </style>
        </head>
        <body>
            {html_text}
        </body>
        </html>
        """

        try:
            # Create PDF using xhtml2pdf
            with open(filename, "w+b") as pdf_file:
                pisa_status = pisa.CreatePDF(styled_html, dest=pdf_file)

            if pisa_status.err:
                print(f"Error creating PDF: {pisa_status.err}")
                return False
            else:
                print(f"Paper successfully written to {filename}")
                return True

        except Exception as e:
            print(f"Error creating PDF with xhtml2pdf: {e}")
            return False

if __name__ == "__main__":
    topic = "TimeSeries Forecasting using Deep Learning"
    pipeline = ResearchPaperPipeline(topic)
    
    # Execute the pipeline steps
    pipeline.fetch_and_parse_pdfs()
    pipeline.build_knowledge_base()
    
    final_paper_md = pipeline.generate_structured_paper()
    print("Paper generated. Converting to PDF...")

    # Save markdown version for debugging
    with open("research_paper.md", "w", encoding="utf-8") as md_file:
        md_file.write(final_paper_md)
    
    # Convert to PDF
    pipeline.output_pdf(final_paper_md, "research_paper.pdf")
    print("Done! Check research_paper.pdf and research_paper.md")