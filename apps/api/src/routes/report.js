import express from 'express';
import { v4 as uuidv4 } from 'uuid';
import { getAsset, getMatches, saveReport, updateAssetStatus } from '../services/firestoreService.js';
import { generateGeminiReport } from '../services/geminiService.js';

const router = express.Router();

router.post('/:assetId', async (req, res) => {
  try {
    const { assetId } = req.params;
    
    // Fetch asset + matches
    const asset = await getAsset(assetId);
    if (!asset) {
      return res.status(404).json({ error: 'Asset not found', assetId });
    }
    
    const matches = await getMatches(assetId);
    
    // Call geminiService
    const reportData = await generateGeminiReport(asset, matches);
    
    // Create base report object
    const report = {
      reportId: uuidv4(),
      assetId,
      summary: reportData.summary || 'Summary unavailable',
      riskLevel: reportData.riskLevel || 'Unknown',
      recommendation: reportData.recommendation || 'No recommendation',
      takedownText: reportData.takedownText || '',
      generatedAt: new Date().toISOString(),
      ...reportData
    };
    
    // Save report
    await saveReport(report);
    
    // Update asset status
    await updateAssetStatus(assetId, 'reported');
    
    res.json(report);
  } catch (error) {
    if (error.message.includes('Gemini service not yet implemented')) {
      return res.status(500).json({ error: "Gemini service not yet implemented", assetId: req.params.assetId });
    }
    console.error('Report route error:', error);
    res.status(500).json({ error: error.message, assetId: req.params.assetId });
  }
});

export default router;