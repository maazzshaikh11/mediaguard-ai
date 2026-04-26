import express from 'express';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import { saveAsset, updateAssetStatus, saveMatches } from '../services/firestoreService.js';
import { runFingerprint } from '../services/aiService.js';

const router = express.Router();
const storage = multer.memoryStorage();
const upload = multer({ storage });

router.post('/', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded', assetId: null });
    }

    const { title, owner, event, description } = req.body;
    const mediaType = req.file.mimetype;
    const assetId = uuidv4();

    const assetData = {
      assetId,
      title: title || 'Untitled',
      owner: owner || 'Unknown',
      event: event || '',
      description: description || '',
      mediaType,
      uploadedAt: new Date().toISOString(),
      status: 'pending'
    };

    // Save initial asset to Firestore with status "pending"
    await saveAsset(assetData);

    // Respond immediately with pending status
    res.json({ assetId, status: 'pending' });

    // Run scan pipeline asynchronously
    setImmediate(async () => {
      try {
        await updateAssetStatus(assetId, 'scanning');
        
        const result = await runFingerprint(assetId, req.file.buffer, mediaType);
        
        if (result.matches && result.matches.length > 0) {
          await saveMatches(result.matches);
          await updateAssetStatus(assetId, 'matched');
        } else {
          await updateAssetStatus(assetId, 'no_match');
        }
      } catch (error) {
        console.error(`Pipeline error for assetId ${assetId}:`, error);
        try {
          await updateAssetStatus(assetId, 'no_match');
        } catch (dbError) {
          console.error(`Status update error for assetId ${assetId}:`, dbError);
        }
      }
    });

  } catch (error) {
    console.error('Analyze route error:', error);
    res.status(500).json({ error: error.message, assetId: null });
  }
});

export default router;