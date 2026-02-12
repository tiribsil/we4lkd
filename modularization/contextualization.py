import json
import re
from typing import Dict, List, Optional

from llm_utils import get_biomedical_llm
from utils import get_logger

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
        # Ask the model to produce a plain-text, evidence-focused explanation.
        # The caller will convert this text into JSON mapping later.
        return f"""[INST] <<SYS>>
    You are a medicinal chemist and clinical pharmacologist. Your task is to provide a neutral, evidence-based contextualization of a compound for a drug discovery pipeline.

    The pipeline has flagged "{compound}" as a potential candidate for "{self.disease}". This connection may be novel/theoretical or potentially non-existent.

    CRITICAL GUIDELINES:
    1. DO NOT invent clinical evidence. If no known relationship exists, describe the theoretical biochemical rationale OR state that a connection is not biologically plausible.
    2. If the compound is clearly contraindicated, toxic, or irrelevant to the disease (e.g., a pesticide or a completely unrelated drug), explicitly state this.
    3. Distinguish clearly between "Approved Use" and "Theoretical Hypothesis."
    <</SYS>>

    Provide a concise, evidence-focused plain-text explanation about "{compound}" in the context of "{self.disease}". Cover the following points in natural language (but return only plain text):
    - Brief identification (what the compound is)
    - Primary clinical or research uses
    - Mechanism of action summary (molecular targets/pathways)
    - Theoretical connection to the disease (if any) or why no plausible connection exists
    - Plausibility assessment (High/Moderate/Low/Speculative/Implausible) and short justification
    - Key safety considerations or contraindications

    Respond with only the explanatory text (no JSON, no markdown, no code fences). [/INST]"""

    def _extract_json(self, text: str) -> str:
        """Extract JSON from raw text output with multiple fallback strategies and attempt to fix truncated JSON."""
        text = text.strip()

        def _attempt_fix(candidate: str) -> Optional[str]:
            # Remove trailing commas before closing braces/brackets
            candidate = re.sub(r",\s*(\}|])", r"\1", candidate)
            # Balance braces by appending missing closing braces
            opens = candidate.count("{")
            closes = candidate.count("}")
            if opens > closes:
                candidate += "}" * (opens - closes)
            # Try to validate
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                return None

        # Strategy 1: Direct JSON object
        if text.startswith("{") and text.endswith("}"):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                fixed = _attempt_fix(text)
                if fixed:
                    return fixed

        start = text.find("{")
        end = text.rfind("}")

        potential_json = None

        if start != -1:
            if end != -1 and end > start:
                potential_json = text[start : end + 1]
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

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                fixed = _attempt_fix(candidate)
                if fixed:
                    return fixed

        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            candidate = json_match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                fixed = _attempt_fix(candidate)
                if fixed:
                    return fixed

        raise ValueError(
            f"Could not extract JSON from output. Raw text: {text[:200]}..."
        )

    def _create_fallback_entry(self, compound: str, raw_output: str) -> Dict:
        """Create a structured entry from unstructured output."""
        return {
            f"{compound}": raw_output[:500],
            "error": "Model did not return a usable explanation",
        }

    def analyze_compound(
        self,
        compound: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> Dict:
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
                repeat_penalty=1.1,
            )

            raw_text = output["choices"][0]["text"].strip()

            if not raw_text:
                self.logger.warning(f"Empty explanation for {compound}")
                return self._create_fallback_entry(compound, "")

            return {"compound_name": compound, "explanation": raw_text}

        except Exception as e:
            return {"compound_name": compound, "error": f"Inference failure: {str(e)}"}

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
                    self.logger.warning(
                        f"   ✗ Error: {result.get('error', 'Unknown error')}"
                    )

        return results

    def export_json(self, results: List[Dict], filepath: Optional[str] = None) -> str:
        """
        Export results to plain-text mapping lines in the format "compound: explanation".

        Args:
            results: List of analysis results
            filepath: File path (optional)

        Returns:
            Formatted text string where each line is "compound: explanation"
        """
        # Default behavior: export structured JSON (list of {compound, explanation, ...})
        json_results = []

        for r in results:
            # Determine compound name
            name = r.get("compound_name") or r.get("compound")
            if not name:
                keys = [
                    k
                    for k in r.keys()
                    if k not in ("explanation", "compound_name", "error")
                ]
                name = keys[0] if keys else "unknown"

            # Determine explanation text
            explanation = r.get("explanation")
            if not explanation:
                if name in r and isinstance(r[name], str):
                    explanation = r[name]
                else:
                    explanation = r.get("error", "")

            explanation = re.sub(r"\s+", " ", (explanation or "")).strip()

            entry = {"compound": name, "explanation": explanation}

            # Preserve error field if present
            if "error" in r:
                entry["error"] = r["error"]

            json_results.append(entry)

        text_output = json.dumps(json_results, ensure_ascii=False, indent=2)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text_output)
            self.logger.info(f"Results exported to: {filepath}")

        return text_output


if __name__ == "__main__":
    # Compounds to analyze
    # @ti só apontar para o arquivo de compostos finais, e o código vai ler os compostos de lá. Por enquanto, vou deixar uma lista hardcoded pra vc testar.
    compounds = [
        "eganelisib",
        "5-aza-2'-deoxycytidine",
        "pelabresib",
        "monobenzone",
        "aspacytarabine",
        "pemigatinib",
        "alvelestat",
        "vactosertib",
        "cefpodoxime",
        "selpercatinib",
        "sardomozide",
        "zelenirstat",
        "sonrotoclax",
        "pralsetinib",
        "mezigdomide",
        "edoxudin",
        "tuspetinib",
        "n-acetylcochinol-o-phosphate",
        "venetoclax",
        "sparsomycin",
    ]

    print("Initializing BioMistral-7B model...")
    module = ContextualizationModule(disease="Acute Myeloid Leukemia", n_ctx=32768)

    results = module.analyze_batch(compounds)
    output_file = "contextualization.json"
    module.export_json(results=results, filepath=output_file)

    successful = sum(1 for r in results if "error" not in r)
    print(f"\nSummary: {successful}/{len(compounds)} compounds analyzed successfully")
