"""Optional original v2 Gemini judge; no API access on import."""
import os
import json

class GeminiJudge:

    def __init__(self, model_name='gemini-2.0-flash'):
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print('❌ Error: GOOGLE_API_KEY environment variable not found.')
            raise RuntimeError('GOOGLE_API_KEY is required')
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name, generation_config={'temperature': 0.0, 'response_mime_type': 'application/json'})
        self.system_prompt = '\n        You are an expert judge evaluating "Memorization".\n        Score from 0.00 to 100.00.\n        Output JSON: {"score": <float>, "reasoning": "..."}\n        '

    def evaluate(self, ground_truth, prediction):
        prompt = f'\n        {self.system_prompt}\n        **Ground Truth:** {ground_truth}\n        **Prediction:** {prediction}\n        Identify key entities in Ground Truth and check if Prediction has memorized them.\n        '
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            return (float(result.get('score', 0.0)), None)
        except Exception as e:
            return (None, str(e))
