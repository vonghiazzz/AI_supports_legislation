import { useState, useRef, useEffect } from 'react';
import { Send, Scale, ShieldAlert, BookOpen, Loader2 } from 'lucide-react';
import Message from './components/Message';
import SourceCard from './components/SourceCard';
import LawModal from './components/LawModal';
import { askLegalAI } from './api';

function App() {
  const [messages, setMessages] = useState([
    {
      role: 'ai',
      content: 'Chào mừng bạn đến với **Hệ thống Trợ lý Thẩm phán AI**. Tôi đã sẵn sàng hỗ trợ bạn phân tích các hồ sơ vụ việc dựa trên Bộ luật Dân sự 2015. Bạn đang gặp tranh chấp hay thắc mắc pháp lý nào?'
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sources, setSources] = useState([]);
  const [selectedLaw, setSelectedLaw] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const data = await askLegalAI(input);
      
      const aiMessage = {
        role: 'ai',
        content: data.answer,
        processing_time: data.processing_time
      };
      
      setMessages(prev => [...prev, aiMessage]);
      setSources(data.sources || []);
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'ai', 
        content: '⚠️ Xin lỗi, hệ thống đang gặp gián đoạn kết nối. Vui lòng thử lại sau.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSourceClick = (law) => {
    setSelectedLaw(law);
    setIsModalOpen(true);
  };

  return (
    <div className="flex h-screen bg-[#020617] text-slate-200 overflow-hidden font-sans">
      {/* Sidebar - CĂN CỨ PHÁP LÝ */}
      <aside className="w-80 border-r border-slate-800 bg-[#0f172a]/50 flex flex-col p-6 overflow-hidden">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-3 rounded-2xl bg-blue-600/20 border border-blue-500/30">
            <Scale className="text-blue-500" size={28} />
          </div>
          <div>
            <h1 className="text-xl font-black tracking-tight text-white uppercase italic">JUDGE AI</h1>
            <p className="text-[10px] text-blue-500 font-bold uppercase tracking-[0.2em]">Sử dụng BLDS 2015</p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
          <div className="flex items-center gap-2 mb-4 text-xs font-bold text-slate-400 uppercase tracking-widest">
            <BookOpen size={14} className="text-slate-600" />
            Căn cứ pháp lý ({sources.length})
          </div>
          
          {sources.length > 0 ? (
            sources.map((src, i) => (
              <SourceCard 
                key={i} 
                source={src} 
                onClick={handleSourceClick} 
              />
            ))
          ) : (
            <div className="glass-panel p-6 border-dashed text-center opacity-40">
              <ShieldAlert className="mx-auto mb-2 text-slate-600" size={24} />
              <p className="text-[10px]">Chưa có tài liệu tham chiếu</p>
            </div>
          )}
        </div>
        
        <div className="mt-6 pt-6 border-t border-slate-800/50">
          <div className="p-4 bg-blue-900/10 rounded-xl border border-blue-500/10">
            <p className="text-[10px] text-blue-400/80 leading-relaxed italic">
              "Công lý không chỉ là trừng phạt, mà là bảo vệ quyền lợi chính đáng của mỗi cá nhân."
            </p>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900/20 via-[#020617] to-[#020617]">
        {/* Messages List */}
        <div className="flex-1 overflow-y-auto px-6 py-8 custom-scrollbar">
          <div className="max-w-3xl mx-auto">
            {messages.map((m, i) => <Message key={i} message={m} />)}
            {isLoading && (
              <div className="flex justify-start mb-6 animate-pulse">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                  <Loader2 className="text-blue-500 animate-spin" size={20} />
                </div>
                <div className="ml-3 chat-bubble-ai bg-slate-800/40 italic text-slate-500 text-xs flex items-center">
                  Thẩm phán đang nghiên cứu hồ sơ và trích lục pháp luật...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-6 bg-gradient-to-t from-[#020617] via-[#020617] to-transparent">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSubmit} className="relative group">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Nhập thông tin vụ việc hoặc yêu cầu pháp lý..."
                className="w-full bg-slate-900/80 border border-slate-700/80 rounded-2xl py-4 pl-6 pr-14 
                           focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20 
                           transition-all placeholder:text-slate-600 text-sm shadow-2xl"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="absolute right-2 top-2 p-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 
                           text-white rounded-xl transition-all shadow-lg active:scale-95 flex items-center justify-center"
              >
                <Send size={18} />
              </button>
            </form>
            <p className="mt-3 text-[10px] text-center text-slate-500 uppercase tracking-[0.15em]">
              AI chỉ mang tính chất tham khảo &middot; Được hỗ trợ bởi Bộ luật Dân sự 2015
            </p>
          </div>
        </div>
      </main>

      {/* Law Detail Modal */}
      <LawModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
        law={selectedLaw} 
      />
    </div>
  );
}

export default App;
