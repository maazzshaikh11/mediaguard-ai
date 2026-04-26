import { getFirestore } from 'firebase-admin/firestore';

export async function saveAsset(data) {
  try {
    const db = getFirestore();
    const docRef = db.collection('assets').doc(data.assetId);
    await docRef.set(data);
  } catch (error) {
    throw new Error(`Failed to save asset: ${error.message}`);
  }
}

export async function getAsset(id) {
  try {
    const db = getFirestore();
    const docRef = db.collection('assets').doc(id);
    const doc = await docRef.get();
    if (!doc.exists) {
      return null;
    }
    return doc.data();
  } catch (error) {
    throw new Error(`Failed to get asset: ${error.message}`);
  }
}

export async function updateAssetStatus(id, status) {
  try {
    const db = getFirestore();
    const docRef = db.collection('assets').doc(id);
    await docRef.update({ status: status });
  } catch (error) {
    throw new Error(`Failed to update asset status: ${error.message}`);
  }
}

export async function saveMatches(matchesArray) {
  if (!matchesArray || matchesArray.length === 0) return;
  try {
    const db = getFirestore();
    const batch = db.batch();
    matchesArray.forEach(match => {
      const docRef = db.collection('matches').doc(match.matchId);
      batch.set(docRef, match);
    });
    await batch.commit();
  } catch (error) {
    throw new Error(`Failed to save matches: ${error.message}`);
  }
}

export async function getMatches(assetId) {
  try {
    const db = getFirestore();
    const querySnapshot = await db.collection('matches')
      .where('assetId', '==', assetId)
      .orderBy('similarityScore', 'desc')
      .get();
    
    const matches = [];
    querySnapshot.forEach(doc => {
      matches.push(doc.data());
    });
    return matches;
  } catch (error) {
    throw new Error(`Failed to get matches: ${error.message}`);
  }
}

export async function saveReport(data) {
  try {
    const db = getFirestore();
    const docRef = db.collection('reports').doc(data.reportId);
    await docRef.set(data);
  } catch (error) {
    throw new Error(`Failed to save report: ${error.message}`);
  }
}