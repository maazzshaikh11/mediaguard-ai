import axios from 'axios';

export async function generateGeminiReport(asset, matches) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not defined");
  }

  const prompt = `You are a copyright and media risk analysis assistant. Analyze the following uploaded file metadata and AI matches to generate a clear summary report.

**Asset Context:**
- Title: ${asset.title || 'Unknown'}
- Owner/Uploader: ${asset.owner || 'Unknown'}
- Description: ${asset.description || 'None'}
- Media Type: ${asset.mediaType || 'Unknown'}

**Matches Found in our Index:**
${JSON.stringify(matches, null, 2)}

Provide a structured JSON output with the following exact keys (no markdown formatting outside of JSON):
{
  "summary": "A concise 1-2 sentence overview of whether there is a risk of infringement.",
  "riskLevel": "Low", "Medium", or "High",
  "recommendation": "What the user should do next (e.g., 'Safe to publish', 'Review manually', 'Takedown advised').",
  "takedownText": "If High risk, provide a short polite takedown DMCA template, otherwise leave empty string."
}`;

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=${apiKey}`;

  try {
    const response = await axios.post(url, {
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { responseMimeType: "application/json" }
    });

    const textOutput = response.data.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!textOutput) {
      throw new Error("No text response from Gemini");
    }

    // Parse the JSON string from Gemini
    return JSON.parse(textOutput);
  } catch (error) {
    console.error('Gemini API Error:', error.response?.data || error.message);
    throw new Error('Failed to generate report via Gemini');
  }
}
