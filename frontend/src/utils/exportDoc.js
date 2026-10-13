import { exportLegalAnalysisDocx } from '../api';

const DEFAULT_FILENAME = 'Bao-cao-phan-tich-phap-ly.docx';

const extractFilename = (contentDisposition) => {
  if (!contentDisposition) {
    return DEFAULT_FILENAME;
  }

  const match = contentDisposition.match(
    /filename="?([^";]+)"?/i
  );

  return match?.[1] || DEFAULT_FILENAME;
};

export const exportToDoc = async (message) => {
  if (!message || message.role !== 'ai' || !message.content?.trim()) {
    return;
  }

  try {
    const response = await exportLegalAnalysisDocx(message.content);

    const filename = extractFilename(
      response.headers?.['content-disposition']
    );

    const url = URL.createObjectURL(response.data);
    const link = document.createElement('a');

    link.href = url;
    link.download = filename;

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Lỗi khi xuất báo cáo DOCX:', error);
    alert(
      'Có lỗi xảy ra khi tạo tệp Word. '
      + 'Vui lòng kiểm tra kết nối với Backend.'
    );
  }
};
