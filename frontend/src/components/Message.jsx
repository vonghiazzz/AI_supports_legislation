import React from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Gavel } from 'lucide-react';

const Message = ({ message }) => {
  const isAI = message.role === 'ai';

  return (
    <div className={`flex w-full mb-6 ${isAI ? 'justify-start' : 'justify-end'}`}>
      <div className={`flex max-w-[85%] ${isAI ? 'flex-row' : 'flex-row-reverse'}`}>
        <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center border ${
          isAI ? 'bg-slate-800 border-slate-700' : 'bg-blue-600 border-blue-500 ml-3 mr-0'
        } ${!isAI && 'mr-0 ml-3'}`}>
          {isAI ? <Gavel size={20} className="text-blue-400" /> : <User size={20} className="text-white" />}
        </div>
        
        <div className={`mx-3 ${isAI ? 'chat-bubble-ai' : 'chat-bubble-user'}`}>
          <div className="prose prose-invert max-w-none text-sm leading-relaxed">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
          
          {isAI && message.processing_time && (
            <div className="mt-3 pt-3 border-t border-slate-700/50 flex items-center gap-2 text-[10px] text-slate-500 uppercase tracking-widest font-semibold">
              <span className="w-2 h-2 rounded-full bg-green-500/50"></span>
              Phân tích trong {message.processing_time}s
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;
