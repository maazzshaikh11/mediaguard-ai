import axios from 'axios';
import FormData from 'form-data';

export async function runFingerprint(assetId, fileBuffer, mediaType) {
  try {
    const aiServiceUrl = process.env.AI_SERVICE_URL;
    if (!aiServiceUrl) {
      throw new Error('AI_SERVICE_URL environment variable is not defined');
    }

    const form = new FormData();
    form.append('assetId', String(assetId));
    form.append('mediaType', String(mediaType));
    
    form.append('file', fileBuffer, {
      filename: 'upload', 
      contentType: mediaType
    });

    const response = await axios.post(`${aiServiceUrl}/fingerprint`, form, {
      headers: {
        ...form.getHeaders()
      }
    });

    return response.data; // Expected { fingerprint, matches }
  } catch (error) {
    if (error.response) {
      throw new Error(`AI Service Error (${error.response.status}): ${JSON.stringify(error.response.data)}`);
    } else {
      throw new Error(`AI Service Request Failed: ${error.message}`);
    }
  }
}