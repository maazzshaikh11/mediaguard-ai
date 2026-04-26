import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import admin from 'firebase-admin';
import multer from 'multer';
import authMiddleware from './middleware/auth.js';
import errorHandler from './middleware/errorHandler.js';
import { validateAnalyzeBody } from './middleware/validate.js';
import analyzeRouter from './routes/analyze.js';
import assetsRouter from './routes/assets.js';
import matchesRouter from './routes/matches.js';
import reportRouter from './routes/report.js';

dotenv.config();

// Initialize Firebase Admin
if (process.env.FIREBASE_SERVICE_ACCOUNT_JSON) {
  try {
    const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON);
    admin.initializeApp({
      credential: admin.credential.cert(serviceAccount)
    });
    console.log('Firebase Admin initialized');
  } catch (error) {
    console.error('Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON', error);
  }
} else {
  console.warn('FIREBASE_SERVICE_ACCOUNT_JSON is not set in environment variables');
}

const app = express();

// Middleware
app.use(helmet());
app.use(cors({ origin: process.env.FRONTEND_URL }));
app.use(express.json());

// Multer setup (memory storage, max 50MB)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }
});

// Routes
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

const apiRouter = express.Router();

apiRouter.use('/analyze', authMiddleware, analyzeRouter);
apiRouter.use('/assets', authMiddleware, assetsRouter);
apiRouter.use('/matches', authMiddleware, matchesRouter);
apiRouter.use('/report', authMiddleware, reportRouter);

app.use('/api', apiRouter);

// Global Error Handler (must be last)
app.use(errorHandler);

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
