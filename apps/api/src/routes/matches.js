import express from 'express';
import { getMatches } from '../services/firestoreService.js';

const router = express.Router();

router.get('/:assetId', async (req, res) => {
  try {
    const { assetId } = req.params;
    const matches = await getMatches(assetId);
    
    res.json({
      assetId,
      matches: matches || []
    });
  } catch (error) {
    console.error('Matches route error:', error);
    res.status(500).json({ error: error.message, assetId: req.params.assetId });
  }
});

export default router;