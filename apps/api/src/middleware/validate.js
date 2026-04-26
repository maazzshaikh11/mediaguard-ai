export const validateAnalyzeBody = (req, res, next) => {
  const requiredFields = ['title', 'owner', 'event', 'description'];
  
  if (!req.file) {
    return res.status(400).json({ error: 'file is required', assetId: null });
  }

  for (const field of requiredFields) {
    if (!req.body[field]) {
      return res.status(400).json({ error: `${field} is required`, assetId: null });
    }
  }

  next();
};
