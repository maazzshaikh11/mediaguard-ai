import express from 'express';
import { getAsset } from '../services/firestoreService.js';

const router = express.Router();

router.get('/:assetId', async (req, res) => {
  try {
    const { assetId } = req.params;
    const asset = await getAsset(assetId);
    
    if (!asset) {
      return res.status(404).json({ error: 'Asset not found', assetId });
    }
    
    res.json(asset);
  } catch (error) {
    console.error('Assets route error:', error);
    res.status(500).json({ error: error.message, assetId: req.params.assetId });
  }
});

export default router;