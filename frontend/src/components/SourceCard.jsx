import React from 'react';
import { BookMarked, ChevronRight } from 'lucide-react';

const SourceCard = ({ source, onClick }) => {
  return (
    <div 
      className="glass-panel p-4 mb-3 hover:border-blue-500/50 transition-all cursor-pointer group hover:bg-slate-800/40"
      onClick={() => onClick(source)}
    >
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-lg bg-slate-800 border border-slate-700 text-blue-400 group-hover:bg-blue-600/20 group-hover:border-blue-500/30 transition-all">
          <BookMarked size={16} />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-xs font-bold text-slate-100 mb-1 truncate group-hover:text-blue-400 transition-colors">
            Điều {source.law_id}: {source.title}
          </h4>
          <p className="text-[11px] text-slate-400 line-clamp-2 italic">
            {source.chapter} / {source.section}
          </p>
        </div>
        <ChevronRight size={14} className="text-slate-600 group-hover:text-blue-400 mt-1 transition-transform group-hover:translate-x-0.5" />
      </div>
    </div>
  );
};

export default SourceCard;
