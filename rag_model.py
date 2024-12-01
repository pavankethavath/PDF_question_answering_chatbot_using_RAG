from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex 
from llama_index.core.retrievers import VectorIndexRetriever 
from llama_index.core.query_engine import RetrieverQueryEngine 
from llama_index.core.postprocessor import SimilarityPostprocessor 
from llama_index.readers.file import PDFReader 
from transformers import AutoModelForCausalLM, AutoTokenizer  
import tempfile 
import os 

class RAGSystem: 
    def __init__(self): 
        self._initialize_settings() 
        self._initialize_model() 
        self.index = None 

    
    def _initialize_settings(self):  

        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") 
        Settings.llm = None
        Settings.chunk_size = 256
        Settings.chunk_overlap = 15  

    def _initialize_model(self): 
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            trust_remote_code = False, 
            revision = "main", 
            # device_map='cuda:0' # We try to load model on the GPU
            )  

        # load tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True) 

    def process_pdf(self, file_content): 

        try: 
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file: 
                tmp_file.write(file_content) 
                tmp_path = tmp_file.name 

        # Read PDF 

            reader = PDFReader() 
            documents = reader.load_data(tmp_path) 

            self.index = VectorStoreIndex.from_documents(documents) 

            #cleanup 
            os.unlike(tmp_path) 

            return True 
        
        except Exception as e: 
            print(f"Error processing PDF: {str(e)}") 
            return False 
        

    def get_query_engine(self, top_k = 2): 

        if not self.index: 
            raise ValueError("No index available. Please Process a PDF first") 
        
        retriever = VectorIndexRetriever(
            index=self.index, 
            similarity_top_k=top_k,
            ) 
        
        return RetrieverQueryEngine(
            retriever=retriever, 
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
            )  
    
    def _create_prompt(self, context, query): 
        return f"""you are an AI assistant tasked with answering question based on the provided PDF content. 
Please analyze the following excerpt from the PDF and answer the question. 
PDF content: 
{context} 

Question: {query} 

Instructions: 
- Answer only based on the information provided in the PDF content above. 
- If the Answer cannot be found in the provided content, say "I cannot find the answer to the question and provide a pdf documents" 
- Be concise and specifice. 
- Include relevant quote or references from the PDF when applicable 

Answer:"""  
    

    def generate_response(self, query_engine, query): 

        try: 
            response = query_engine.query(query)  

            context = ""  

            for node in response.source_nodes[:2]: 
                context += f"{node.text}\n\n" 

            if not context.strip(): 
                return "No relvant information from PDF document" 
            
            # Creating a prompt and generatig a response 
            prompt = self._create_prompt(context, query) 

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True) 
            outputs = self.model.generate(
                input_ids = inputs['input_ids'], 
                max_new_tokens = 512, 
                num_return_sequences = 1, 
                temperature = 0.3, 
                top_p = 0.9, 
                do_sample =True, 
                repetition_penalty = 1.2 
            )  

            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)   

            if "Answer" in response_text: 
                response_text = response_text.split("Answer: ")[-1].strip() 

            return response_text if response_text else "Unable to generate a response from PDF documents" 
    

        except Exception as e:  
            print(f"Error generating a responde: {str(e)}") 
            return f"Error processing your question: {str(e)}"



