import axios from 'axios';

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
).replace(/\/+$/, '');

export const askLegalAI = async (query, topK = 5) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/ask`, {
      query,
      top_k: topK
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const searchLegalDocs = async (query, topK = 5) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/search`, {
      query,
      top_k: topK
    });
    return response.data;
  } catch (error) {
    console.error('Search API Error:', error);
    throw error;
  }
};

export const exportLegalAnalysisDocx = async (content) => {
  try {
    return await axios.post(
      `${API_BASE_URL}/ai/export-docx`,
      { content },
      { responseType: 'blob' }
    );
  } catch (error) {
    console.error('DOCX Export API Error:', error);
    throw error;
  }
};
