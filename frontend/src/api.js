import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

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
