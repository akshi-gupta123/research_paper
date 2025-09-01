import arxiv
import requests
import PyPDF2
import io
from openai import OpenAI
import re
from typing import List, Dict, Tuple
import markdown
import os
from dotenv import load_dotenv
from xhtml2pdf import pisa

load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ResearchPaperPipeline:
    def __init__(self, topic: str):
        self.topic = topic
        self.papers = []  # Stores arxiv result objects
        self.full_texts = []  # Stores extracted text for each paper
        self.knowledge_base = []  # Stores chunks of text with metadata
        self.references = []  # Stores reference information

    def search_literature(self, max_results: int = 5):
        """Searches arXiv for papers related to the topic."""
        search = arxiv.Search(
            query=self.topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        self.papers = list(search.results())
        print(f"Found {len(self.papers)} papers on arXiv.")

    def fetch_and_parse_pdfs(self):
        """Downloads and extracts text from the found PDFs."""
        for paper in self.papers:
            try:
                response = requests.get(paper.pdf_url)
                response.raise_for_status()

                with io.BytesIO(response.content) as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    # Store paper text and metadata
                    self.full_texts.append({
                        'title': paper.title,
                        'authors': [a.name for a in paper.authors],
                        'published': paper.published.strftime("%Y-%m-%d"),
                        'summary': paper.summary,
                        'full_text': text,
                        'url': paper.pdf_url,
                        'entry_id': paper.entry_id
                    })
                    print(f"Fetched and parsed: {paper.title}")
            except Exception as e:
                print(f"Failed to process {paper.title}: {e}")

    def build_knowledge_base(self, chunk_size: int = 300):
        """Chunks the full text of papers into smaller, citable sentences/paragraphs."""
        for paper_info in self.full_texts:
            text = paper_info['full_text']
            # Simple chunking by sentence-ending punctuation followed by space and capital letter.
            chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            for chunk in chunks:
                if len(chunk) > 50:  # Filter out very short chunks
                    self.knowledge_base.append({
                        'text': chunk,
                        'source_title': paper_info['title'],
                        'source_authors': paper_info['authors'],
                        'source_url': paper_info['url'],
                        'source_id': paper_info['entry_id']
                    })
        print(f"Built knowledge base with {len(self.knowledge_base)} citable chunks.")

    def generate_structured_paper(self) -> str:
        """Uses an LLM to generate the research paper by querying the knowledge base."""
        # 2. For each section, find relevant citations and generate content.
        section_heads = ["Abstract", "Introduction", "Related Work", "Methodology", "Results", "Discussion", "Conclusion"]
        paper_markdown = f"# Research Paper: {self.topic}\n\n"
        paper_content = []

        for section in section_heads:
            # Find the most relevant chunks for this specific section
            # In a real system, you would use a vector DB & embedding similarity for this step.
            # Here we just take a few random chunks for demonstration.
            relevant_chunks = self.knowledge_base[:3]

            # Build a prompt for the LLM to write the section using the citations.
            section_prompt = f"""
            Write the **{section}** section for a two-page research paper on **{self.topic}**.

            You must ground every claim you make with direct evidence from the provided research excerpts. Weave the citations seamlessly into the narrative.

            Instructions:
            1. Use the following excerpts from research papers as your sole source of evidence.
            2. For every factual statement, cite the source using the provided ID in brackets, e.g., [1].
            3. Write authoritatively and concisely. This section should be roughly 2-4 paragraphs.

            Excerpts:
            {self._format_chunks_for_prompt(relevant_chunks)}

            Begin writing the {section} section now:
            """

            section_text = self._query_llm(section_prompt)
            paper_markdown += f"## {section}\n\n{section_text}\n\n"

            # Store the references used in this section
            for chunk in relevant_chunks:
                if chunk not in self.references:
                    self.references.append(chunk)

        # 3. Add a references section
        paper_markdown += "## References\n\n"
        for i, ref in enumerate(self.references):
            paper_markdown += f"{[i+1]} {ref['source_title']} by {', '.join(ref['source_authors'][:2])}. {ref['source_url']}\n"

        return paper_markdown

    def _format_chunks_for_prompt(self, chunks: List[Dict]) -> str:
        """Formats knowledge base chunks for the LLM prompt, assigning temporary IDs."""
        formatted_text = ""
        for idx, chunk in enumerate(chunks):
            formatted_text += f"[Source ID: {idx+1}]: {chunk['text']}\n\n"
        return formatted_text

    def _query_llm(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Helper function to query the OpenAI API."""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating content: {e}"

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

# --- Run the Pipeline ---
if __name__ == "__main__":
    topic = "Contrastive Learning in Computer Vision"
    pipeline = ResearchPaperPipeline(topic)
    
    # Execute the pipeline steps
    pipeline.search_literature(max_results=3)
    pipeline.fetch_and_parse_pdfs()
    pipeline.build_knowledge_base()
    
    final_paper_md = pipeline.generate_structured_paper()
    print("Paper generated. Converting to PDF...")
    
    # Save markdown version for debugging
    with open("research_paper.md", "w") as md_file:
        md_file.write(final_paper_md)
    
    # Convert to PDF
    pipeline.output_pdf(final_paper_md, "research_paper.pdf")
    print("Done! Check research_paper.pdf and research_paper.md")