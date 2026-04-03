import React from 'react';
import { X, BookOpen, Scale, FileText } from 'lucide-react';

const LawModal = ({ isOpen, onClose, law }) => {
  if (!isOpen || !law) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-300" 
        onClick={onClose}
      />
      
      {/* Modal Content */}
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-3xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-300">
        {/* Header */}
        <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-2xl bg-blue-600/20 border border-blue-500/30 text-blue-400">
              <Scale size={24} />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Điều {law.law_id}</h2>
              <p className="text-xs text-slate-400 uppercase tracking-widest font-semibold">{law.chapter}</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-slate-800 rounded-full transition-colors text-slate-400 hover:text-white"
          >
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="p-8 max-h-[70vh] overflow-y-auto custom-scrollbar">
          <div className="mb-6 flex items-center gap-2 text-blue-400">
            <BookOpen size={18} />
            <span className="text-sm font-bold uppercase tracking-tight">{law.title}</span>
          </div>
          
          <div className="bg-slate-950/50 p-6 rounded-2xl border border-slate-800/50 leading-relaxed text-slate-300 text-sm whitespace-pre-wrap">
            {law.content}
          </div>

          <div className="mt-6 flex flex-wrap gap-2">
            {law.tags && law.tags.map((tag, i) => (
              <span key={i} className="px-3 py-1 bg-slate-800 text-slate-400 text-[10px] font-bold rounded-full border border-slate-700 uppercase">
                #{tag}
              </span>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-slate-900/80 border-t border-slate-800 flex justify-end">
          <button 
            onClick={onClose}
            className="px-6 py-2 bg-slate-100 hover:bg-white text-slate-950 text-xs font-bold rounded-xl transition-all active:scale-95"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
};

export default LawModal;
