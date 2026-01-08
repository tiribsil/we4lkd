import os
import json
import re
from typing import List, Dict, Optional
from utils import get_logger
from llm_utils import get_biomedical_llm

# Setup Instructions:
# pip install llama-cpp-python
# pip install huggingface-hub

class ContextualizationModule:
    """Medical compound analyzer using BioMistral-7B (CPU-Only)."""
    
    def __init__(self, disease: str, n_ctx: int = 32768):
        """
        Initialize the contextualization module with a GGUF model.
        
        Args:
            n_ctx: Context size
        """
        self.logger = get_logger(self.__class__.__name__)
        self.llm = get_biomedical_llm(n_ctx=n_ctx)
        self.disease = disease
    
    def _create_prompt(self, compound: str) -> str:
        """Generate structured prompt optimized for BioMistral."""
        return f"""[INST] <<SYS>>
You are a medicinal chemist and clinical pharmacologist. Your task is to provide a neutral, evidence-based contextualization of a compound for a drug discovery pipeline. 

The pipeline has flagged "{compound}" as a potential candidate for "{self.disease}". This connection may be novel/theoretical or potentially non-existent. 

**CRITICAL GUIDELINES:**
1. DO NOT invent clinical evidence. If no known relationship exists, describe the theoretical biochemical rationale OR state that a connection is not biologically plausible.
2. If the compound is clearly contraindicated, toxic, or irrelevant to the disease (e.g., a pesticide or a completely unrelated drug), explicitly state this in the 'rationale' field.
3. Distinguish clearly between "Approved Use" and "Theoretical Hypothesis."
<</SYS>>

Provide information about "{compound}" regarding "{self.disease}" in the following JSON format only:

{{
  "compound_name": "string",
  "cas_registry_number": "string",
  "pharmacological_class": "e.g., Small molecule, Monoclonal antibody, etc.",
  "primary_clinical_use": "Current FDA-approved or common research uses",
  "mechanism_of_action_summary": "Biochemical pathway and molecular targets",
  "theoretical_connection_to_disease": "Describe the potential biochemical or molecular rationale for why this might treat {self.disease}. If the connection is based on a specific pathway, name it.",
  "plausibility_assessment": "Select one: [High / Moderate / Low / Speculative / Implausible]",
  "safety_and_risks": "Briefly list major contraindications or reasons why this compound might be unsuitable for {self.disease}."
}}

Respond with only the JSON object. [/INST]"""
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from raw text output with multiple fallback strategies and attempt to fix truncated JSON."""
        text = text.strip()
        
        def _attempt_fix(candidate: str) -> Optional[str]:
            # Remove trailing commas before closing braces/brackets
            candidate = re.sub(r',\s*(\}|])', r'\1', candidate)
            # Balance braces by appending missing closing braces
            opens = candidate.count('{')
            closes = candidate.count('}')
            if opens > closes:
                candidate += '}' * (opens - closes)
            # Try to validate
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                return None

        # Strategy 1: Direct JSON object
        if text.startswith('{') and text.endswith('}'):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                fixed = _attempt_fix(text)
                if fixed:
                    return fixed
        
        start = text.find('{')
        end = text.rfind('}')
        
        potential_json = None
        
        if start != -1:
            if end != -1 and end > start:
                potential_json = text[start:end + 1]
            else:
                potential_json = text[start:]
                
            if potential_json:
                try:
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    fixed = _attempt_fix(potential_json)
                    if fixed:
                        return fixed
        
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                fixed = _attempt_fix(candidate)
                if fixed:
                    return fixed
        
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            candidate = json_match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                fixed = _attempt_fix(candidate)
                if fixed:
                    return fixed
        
        raise ValueError(f"Could not extract JSON from output. Raw text: {text[:200]}...")
    
    def _create_fallback_entry(self, compound: str, raw_output: str) -> Dict:
        """Create a structured entry from unstructured output."""
        return {
            "compound_name": compound,
            "cas_registry_number": "Unknown",
            "pharmacological_class": "Analysis failed - check raw output",
            "primary_clinical_use": "Analysis failed - check raw output",
            "mechanism_of_action_summary": "Analysis failed - check raw output",
            "raw_output": raw_output[:500],
            "error": "Failed to extract structured JSON"
        }
    
    def analyze_compound(self, compound: str, max_tokens: int = 2048,
                        temperature: float = 0.1, top_p: float = 0.95) -> Dict:
        """
        Analyze a single medical compound.
        
        Args:
            compound: Name of the compound to analyze
            max_tokens: Maximum number of tokens in response
            temperature: Temperature for generation (lower = more factual)
            top_p: Nucleus sampling parameter
        
        Returns:
            Dictionary with compound analysis
        """
        prompt = self._create_prompt(compound)
        
        try:
            output = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["[/INST]"],
                echo=False,
                repeat_penalty=1.1
            )
            
            raw_text = output['choices'][0]['text'].strip()
            
            try:
                json_str = self._extract_json(raw_text)
                result = json.loads(json_str)
                
                if "compound_name" not in result or not result["compound_name"]:
                    result["compound_name"] = compound
                
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"JSON parsing failed for {compound}")
                self.logger.debug(f"Raw output preview: {raw_text[:200]}")
                return self._create_fallback_entry(compound, raw_text)
            
        except Exception as e:
            return {
                "compound_name": compound,
                "error": f"Inference failure: {str(e)}"
            }
    
    def analyze_batch(self, compounds: List[str], verbose: bool = True) -> List[Dict]:
        """
        Analyze multiple compounds.
        
        Args:
            compounds: List of compound names
            verbose: If True, print progress
        
        Returns:
            List of dictionaries with analyses
        """
        results = []
        
        if verbose:
            self.logger.info(f"Starting analysis of {len(compounds)} compounds...")
        
        for i, compound in enumerate(compounds, 1):
            if verbose:
                self.logger.info(f"[{i}/{len(compounds)}] Processing: {compound}...")
            
            result = self.analyze_compound(compound)
            results.append(result)
            
            if verbose:
                if "error" not in result:
                    self.logger.info(f"   ✓ Analysis completed successfully")
                else:
                    self.logger.warning(f"   ✗ Error: {result.get('error', 'Unknown error')}")
        
        return results
    
    def export_json(self, results: List[Dict], filepath: Optional[str] = None) -> str:
        """
        Export results to formatted JSON.
        
        Args:
            results: List of analysis results
            filepath: File path (optional)
        
        Returns:
            Formatted JSON string
        """
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            self.logger.info(f"Results exported to: {filepath}")
        
        return json_str


if __name__ == "__main__":
    # Compounds to analyze
    compounds = [
        "daunorubicin",
        "thioguanine",
        "vincristine",
        "vincristine",
        "prednisolone",
        "doxorubicin",
        "prednisone",
        "gold",
        "methotrexate",
        "penicillamine",
        "cyclophosphamide",
        "methylprednisolone",
        "5-fluorouracil",
        "calcium",
        "inosine",
        "cortisol",
        "ribavirin",
        "adenine",
        "2-deoxyglucose",
        "vidarabine",
        "uracil"
    ]

    print("Initializing BioMistral-7B model...")
    module = ContextualizationModule(disease= "Acute Myeloid Leukemia",
                                     n_ctx=32768) 
    
    results = module.analyze_batch(compounds)
    output_file = "contextualization.json"
    module.export_json(results=results, filepath=output_file)
    
    successful = sum(1 for r in results if "error" not in r)
    print(f"\nSummary: {successful}/{len(compounds)} compounds analyzed successfully")
