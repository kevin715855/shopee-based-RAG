import { useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import {
  BarChart3,
  Bot,
  CheckCircle2,
  ChevronRight,
  Database,
  Filter,
  Gauge,
  Layers3,
  MessageSquareText,
  Search,
  Send,
  ShieldAlert,
  Sparkles,
  Star,
} from 'lucide-react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

type Product = {
  id: string
  name: string
  category: string
  reviewCount: number
  avgRating: number
}

type Review = {
  id: string
  productId: string
  text: string
  rating: number
  date: string
  source: 'Shopee' | 'Tiki'
  similarity: number
  rerank: number
  tags: string[]
}

type ProductSummary = {
  ratingDistribution: { rating: string; count: number }[]
  sentimentTrend: { month: string; positive: number; neutral: number; negative: number }[]
  pros: string[]
  cons: string[]
  topics: { name: string; count: number }[]
}

type RagAnswer = {
  question: string
  conclusion: string
  positives: string[]
  cautions: string[]
  citations: string[]
  latency: number
  model: string
  embeddingModel: string
  promptContext: string
}

const products: Product[] = [
  {
    id: 'SP-EL-1024',
    name: 'Tai nghe Bluetooth chống ồn AirBass Pro',
    category: 'Đồ điện tử',
    reviewCount: 8421,
    avgRating: 4.6,
  },
  {
    id: 'SP-GD-2031',
    name: 'Máy hút bụi cầm tay HomeClean V2',
    category: 'Gia dụng',
    reviewCount: 5368,
    avgRating: 4.4,
  },
  {
    id: 'SP-EL-4110',
    name: 'Pin sạc dự phòng PowerMax 20000mAh',
    category: 'Phụ kiện điện tử',
    reviewCount: 9734,
    avgRating: 4.7,
  },
]

const summaries: Record<string, ProductSummary> = {
  'SP-EL-1024': {
    ratingDistribution: [
      { rating: '5 sao', count: 5260 },
      { rating: '4 sao', count: 1940 },
      { rating: '3 sao', count: 760 },
      { rating: '2 sao', count: 280 },
      { rating: '1 sao', count: 181 },
    ],
    sentimentTrend: [
      { month: 'T1', positive: 72, neutral: 20, negative: 8 },
      { month: 'T2', positive: 76, neutral: 17, negative: 7 },
      { month: 'T3', positive: 74, neutral: 18, negative: 8 },
      { month: 'T4', positive: 79, neutral: 15, negative: 6 },
      { month: 'T5', positive: 81, neutral: 13, negative: 6 },
    ],
    pros: ['Âm thanh rõ', 'Chống ồn ổn trong tầm giá', 'Pin đủ dùng cả ngày'],
    cons: ['Mic bị rè ở nơi nhiều gió', 'Một số đơn giao thiếu nút tai phụ'],
    topics: [
      { name: 'Pin', count: 1240 },
      { name: 'Âm thanh', count: 1870 },
      { name: 'Mic', count: 920 },
      { name: 'Giao hàng', count: 760 },
      { name: 'Bảo hành', count: 310 },
    ],
  },
  'SP-GD-2031': {
    ratingDistribution: [
      { rating: '5 sao', count: 3020 },
      { rating: '4 sao', count: 1360 },
      { rating: '3 sao', count: 610 },
      { rating: '2 sao', count: 260 },
      { rating: '1 sao', count: 118 },
    ],
    sentimentTrend: [
      { month: 'T1', positive: 68, neutral: 22, negative: 10 },
      { month: 'T2', positive: 71, neutral: 19, negative: 10 },
      { month: 'T3', positive: 70, neutral: 20, negative: 10 },
      { month: 'T4', positive: 73, neutral: 18, negative: 9 },
      { month: 'T5', positive: 75, neutral: 17, negative: 8 },
    ],
    pros: ['Nhẹ tay', 'Hút bụi mịn tốt', 'Dễ tháo hộp bụi'],
    cons: ['Hơi ồn ở mức cao', 'Pin giảm nhanh khi hút turbo'],
    topics: [
      { name: 'Lực hút', count: 1120 },
      { name: 'Pin', count: 830 },
      { name: 'Độ ồn', count: 690 },
      { name: 'Phụ kiện', count: 430 },
      { name: 'Vệ sinh', count: 720 },
    ],
  },
  'SP-EL-4110': {
    ratingDistribution: [
      { rating: '5 sao', count: 6680 },
      { rating: '4 sao', count: 2050 },
      { rating: '3 sao', count: 610 },
      { rating: '2 sao', count: 250 },
      { rating: '1 sao', count: 144 },
    ],
    sentimentTrend: [
      { month: 'T1', positive: 77, neutral: 16, negative: 7 },
      { month: 'T2', positive: 80, neutral: 14, negative: 6 },
      { month: 'T3', positive: 78, neutral: 15, negative: 7 },
      { month: 'T4', positive: 82, neutral: 12, negative: 6 },
      { month: 'T5', positive: 84, neutral: 11, negative: 5 },
    ],
    pros: ['Dung lượng đúng kỳ vọng', 'Sạc nhanh ổn định', 'Vỏ chắc chắn'],
    cons: ['Khá nặng khi mang túi nhỏ', 'Một số người cần tự mua thêm cáp C-C'],
    topics: [
      { name: 'Sạc nhanh', count: 1610 },
      { name: 'Dung lượng', count: 1480 },
      { name: 'Trọng lượng', count: 680 },
      { name: 'Đóng gói', count: 540 },
      { name: 'Bảo hành', count: 430 },
    ],
  },
}

const reviews: Review[] = [
  {
    id: 'RV-8012',
    productId: 'SP-EL-1024',
    text: 'Pin dùng được khoảng 7 tiếng nếu bật chống ồn, âm thanh rõ và bass vừa. Điểm trừ là mic gọi ngoài đường hơi rè.',
    rating: 4,
    date: '2026-04-12',
    source: 'Shopee',
    similarity: 0.91,
    rerank: 0.88,
    tags: ['pin', 'mic', 'âm thanh'],
  },
  {
    id: 'RV-8018',
    productId: 'SP-EL-1024',
    text: 'Mua làm quà khá ổn vì hộp đẹp, tai nghe nhẹ. Shop giao nhanh nhưng thiếu một bộ nút tai phụ.',
    rating: 4,
    date: '2026-03-28',
    source: 'Shopee',
    similarity: 0.83,
    rerank: 0.79,
    tags: ['giao hàng', 'đóng gói', 'quà tặng'],
  },
  {
    id: 'RV-8026',
    productId: 'SP-EL-1024',
    text: 'Chống ồn dùng trong văn phòng tốt, đi xe máy thì vẫn nghe tiếng gió. Kết nối điện thoại nhanh, chưa thấy lỗi.',
    rating: 5,
    date: '2026-02-20',
    source: 'Tiki',
    similarity: 0.89,
    rerank: 0.84,
    tags: ['chống ồn', 'kết nối'],
  },
  {
    id: 'RV-5102',
    productId: 'SP-GD-2031',
    text: 'Máy nhẹ, hút tóc và bụi ghế sofa tốt. Chế độ turbo khỏe nhưng pin tụt nhanh sau khoảng 15 phút.',
    rating: 4,
    date: '2026-04-02',
    source: 'Shopee',
    similarity: 0.88,
    rerank: 0.85,
    tags: ['lực hút', 'pin'],
  },
  {
    id: 'RV-5114',
    productId: 'SP-GD-2031',
    text: 'Hộp bụi tháo dễ, vệ sinh nhanh. Tiếng ồn hơi lớn nếu dùng ban đêm, còn lại ổn so với giá.',
    rating: 4,
    date: '2026-03-16',
    source: 'Tiki',
    similarity: 0.82,
    rerank: 0.81,
    tags: ['vệ sinh', 'độ ồn'],
  },
  {
    id: 'RV-9301',
    productId: 'SP-EL-4110',
    text: 'Sạc iPhone và tai nghe cùng lúc vẫn ổn, máy không nóng nhiều. Dung lượng dùng được gần hai ngày đi công tác.',
    rating: 5,
    date: '2026-04-21',
    source: 'Shopee',
    similarity: 0.92,
    rerank: 0.9,
    tags: ['sạc nhanh', 'dung lượng'],
  },
  {
    id: 'RV-9320',
    productId: 'SP-EL-4110',
    text: 'Pin chắc tay nhưng hơi nặng. Đóng gói kỹ, có cáp USB-A đi kèm, muốn sạc nhanh laptop thì phải mua thêm cáp C-C.',
    rating: 4,
    date: '2026-03-06',
    source: 'Tiki',
    similarity: 0.86,
    rerank: 0.82,
    tags: ['trọng lượng', 'phụ kiện', 'đóng gói'],
  },
]

const sampleQuestions = [
  'Pin có tốt không?',
  'Sản phẩm có hay lỗi không?',
  'Có phù hợp làm quà không?',
  'Điểm trừ phổ biến là gì?',
]

const pipelineSteps = ['Embedding', 'Retrieval', 'Rerank', 'LLM']

function buildAnswer(question: string, product: Product, selectedReviews: Review[], useRerank: boolean): RagAnswer {
  const topReviews = [...selectedReviews]
    .sort((a, b) => (useRerank ? b.rerank - a.rerank : b.similarity - a.similarity))
    .slice(0, 3)

  if (topReviews.length === 0) {
    return {
      question,
      conclusion: 'Không tìm thấy đủ bằng chứng trong đánh giá đã thu thập để trả lời chắc chắn.',
      positives: [],
      cautions: ['Hãy nới bộ lọc rating, nguồn dữ liệu hoặc thử câu hỏi cụ thể hơn.'],
      citations: [],
      latency: 420,
      model: 'gpt-4o-mini',
      embeddingModel: 'bge-m3',
      promptContext: 'Không có review phù hợp sau bước lọc.',
    }
  }

  const hasBatteryIntent = /pin|sạc|dung lượng/i.test(question)
  const hasGiftIntent = /quà|tặng|hộp/i.test(question)
  const hasIssueIntent = /lỗi|điểm trừ|vấn đề|hỏng/i.test(question)

  const conclusion = hasBatteryIntent
    ? `${product.name} được đánh giá tích cực về pin/dung lượng trong các review liên quan, nhưng nên lưu ý hiệu năng giảm khi dùng chế độ nặng.`
    : hasGiftIntent
      ? `${product.name} khá phù hợp để làm quà nếu ưu tiên đóng gói và trải nghiệm nhận hàng, nhưng cần kiểm tra phụ kiện đi kèm.`
      : hasIssueIntent
        ? `Các lỗi phổ biến không xuất hiện dày đặc trong nhóm review truy xuất, nhưng có một số phàn nàn lặp lại về phụ kiện, tiếng ồn hoặc mic tùy sản phẩm.`
        : `${product.name} có xu hướng được đánh giá tốt, với điểm mạnh nổi bật nằm ở trải nghiệm sử dụng thực tế hơn là thông số quảng cáo.`

  return {
    question,
    conclusion,
    positives: [
      'Nhiều review có rating 4-5 sao và mô tả trải nghiệm sử dụng cụ thể.',
      'Các ý kiến tích cực xuất hiện nhất quán ở nhóm review có điểm rerank cao.',
    ],
    cautions: [
      'Vẫn có một số điểm trừ cần kiểm tra trước khi mua, đặc biệt là phụ kiện hoặc hiệu năng trong điều kiện sử dụng nặng.',
      'Câu trả lời chỉ dựa trên các review đã crawl và bộ lọc hiện tại.',
    ],
    citations: topReviews.map((review) => review.id),
    latency: useRerank ? 1180 : 760,
    model: 'gpt-4o-mini',
    embeddingModel: 'bge-m3',
    promptContext: topReviews.map((review) => `[${review.id}] ${review.text}`).join('\n'),
  }
}

function App() {
  const [selectedProductId, setSelectedProductId] = useState(products[0].id)
  const [question, setQuestion] = useState('Pin có tốt không?')
  const [ratingFilter, setRatingFilter] = useState('all')
  const [sourceFilter, setSourceFilter] = useState('all')
  const [keyword, setKeyword] = useState('')
  const [useRerank, setUseRerank] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [activeCitation, setActiveCitation] = useState<string | null>(null)
  const [answer, setAnswer] = useState<RagAnswer | null>(() =>
    buildAnswer('Pin có tốt không?', products[0], reviews.filter((review) => review.productId === products[0].id), true),
  )

  const evidenceRefs = useRef<Record<string, HTMLDivElement | null>>({})

  const selectedProduct = products.find((product) => product.id === selectedProductId) ?? products[0]
  const summary = summaries[selectedProduct.id]

  const filteredReviews = useMemo(() => {
    return reviews
      .filter((review) => review.productId === selectedProduct.id)
      .filter((review) => ratingFilter === 'all' || review.rating === Number(ratingFilter))
      .filter((review) => sourceFilter === 'all' || review.source === sourceFilter)
      .filter((review) => {
        const normalizedKeyword = keyword.trim().toLowerCase()
        if (!normalizedKeyword) return true
        return (
          review.text.toLowerCase().includes(normalizedKeyword) ||
          review.tags.some((tag) => tag.toLowerCase().includes(normalizedKeyword))
        )
      })
      .sort((a, b) => (useRerank ? b.rerank - a.rerank : b.similarity - a.similarity))
  }, [keyword, ratingFilter, selectedProduct.id, sourceFilter, useRerank])

  const selectedCitationReviews = useMemo(() => {
    if (!answer) return []
    return answer.citations
      .map((citation) => reviews.find((review) => review.id === citation))
      .filter((review): review is Review => Boolean(review))
  }, [answer])

  const handleAsk = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setIsLoading(true)
    window.setTimeout(() => {
      setAnswer(buildAnswer(question, selectedProduct, filteredReviews, useRerank))
      setIsLoading(false)
    }, 850)
  }

  const handleCitationClick = (reviewId: string) => {
    setActiveCitation(reviewId)
    evidenceRefs.current[reviewId]?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Sparkles size={20} />
          </div>
          <div>
            <p className="eyebrow">Review-based RAG</p>
            <h1>ShopeeFeed</h1>
          </div>
        </div>

        <section className="panel product-panel">
          <div className="panel-title">
            <Database size={18} />
            <h2>Sản phẩm</h2>
          </div>
          <div className="product-list">
            {products.map((product) => (
              <button
                className={`product-button ${product.id === selectedProduct.id ? 'active' : ''}`}
                key={product.id}
                onClick={() => {
                  setSelectedProductId(product.id)
                  setAnswer(buildAnswer(question, product, reviews.filter((review) => review.productId === product.id), useRerank))
                }}
                type="button"
              >
                <span>{product.name}</span>
                <small>
                  {product.category} · {product.reviewCount.toLocaleString('vi-VN')} reviews
                </small>
                <b>
                  <Star size={14} fill="currentColor" /> {product.avgRating}
                </b>
              </button>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-title">
            <Filter size={18} />
            <h2>Bộ lọc truy xuất</h2>
          </div>
          <label>
            Rating
            <select value={ratingFilter} onChange={(event) => setRatingFilter(event.target.value)}>
              <option value="all">Tất cả rating</option>
              <option value="5">5 sao</option>
              <option value="4">4 sao</option>
              <option value="3">3 sao</option>
              <option value="2">2 sao</option>
              <option value="1">1 sao</option>
            </select>
          </label>
          <label>
            Nguồn
            <select value={sourceFilter} onChange={(event) => setSourceFilter(event.target.value)}>
              <option value="all">Shopee + Tiki</option>
              <option value="Shopee">Shopee</option>
              <option value="Tiki">Tiki</option>
            </select>
          </label>
          <label>
            Tìm trong review
            <div className="search-field">
              <Search size={16} />
              <input
                onChange={(event) => setKeyword(event.target.value)}
                placeholder="pin, mic, giao hàng..."
                value={keyword}
              />
            </div>
          </label>
          <label className="switch-row">
            <span>
              <b>BGE Reranker</b>
              <small>Sắp xếp lại top-k review</small>
            </span>
            <input checked={useRerank} onChange={(event) => setUseRerank(event.target.checked)} type="checkbox" />
          </label>
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Demo bảo vệ môn học</p>
            <h2>Phân tích và tóm tắt đánh giá sản phẩm thương mại điện tử</h2>
          </div>
          <div className="api-status">
            <CheckCircle2 size={16} />
            Mock API ready
          </div>
        </header>

        <section className="insights-grid">
          <div className="metric-card">
            <span>Rating trung bình</span>
            <strong>{selectedProduct.avgRating}/5</strong>
            <small>{selectedProduct.reviewCount.toLocaleString('vi-VN')} reviews đã crawl</small>
          </div>
          <div className="metric-card">
            <span>Model trả lời</span>
            <strong>{answer?.model ?? 'gpt-4o-mini'}</strong>
            <small>Embedding: {answer?.embeddingModel ?? 'bge-m3'}</small>
          </div>
          <div className="metric-card">
            <span>Top-k evidence</span>
            <strong>{filteredReviews.length}</strong>
            <small>{useRerank ? 'Đã bật rerank' : 'Chỉ similarity search'}</small>
          </div>
        </section>

        <section className="dashboard-row">
          <article className="panel chart-panel">
            <div className="panel-title">
              <BarChart3 size={18} />
              <h2>Phân bố sao</h2>
            </div>
            <ResponsiveContainer height={168} width="100%">
              <BarChart data={summary.ratingDistribution}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="rating" tickLine={false} />
                <YAxis hide />
                <Tooltip />
                <Bar dataKey="count" fill="#ee4d2d" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </article>

          <article className="panel chart-panel">
            <div className="panel-title">
              <Gauge size={18} />
              <h2>Sentiment theo tháng</h2>
            </div>
            <ResponsiveContainer height={168} width="100%">
              <AreaChart data={summary.sentimentTrend}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="month" tickLine={false} />
                <YAxis hide />
                <Tooltip />
                <Area dataKey="positive" fill="#18a058" fillOpacity={0.18} stroke="#18a058" />
                <Area dataKey="negative" fill="#d03050" fillOpacity={0.12} stroke="#d03050" />
              </AreaChart>
            </ResponsiveContainer>
          </article>
        </section>

        <section className="qa-grid">
          <article className="panel qa-panel">
            <div className="panel-title">
              <MessageSquareText size={18} />
              <h2>Hỏi đáp RAG theo review thật</h2>
            </div>

            <div className="question-chips">
              {sampleQuestions.map((sample) => (
                <button key={sample} onClick={() => setQuestion(sample)} type="button">
                  {sample}
                </button>
              ))}
            </div>

            <form className="ask-box" onSubmit={handleAsk}>
              <textarea
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Nhập câu hỏi về trải nghiệm thực tế của khách hàng..."
                value={question}
              />
              <button disabled={isLoading || !question.trim()} type="submit">
                <Send size={17} />
                Hỏi RAG
              </button>
            </form>

            <div className="pipeline">
              {pipelineSteps.map((step, index) => (
                <div className={`pipeline-step ${isLoading || answer ? 'done' : ''}`} key={step}>
                  <span>{index + 1}</span>
                  {step}
                  {index < pipelineSteps.length - 1 && <ChevronRight size={15} />}
                </div>
              ))}
            </div>

            <section className="answer-box">
              {isLoading ? (
                <div className="loading-state">
                  <Bot size={22} />
                  Đang embedding query, retrieval top-k, rerank evidence và tổng hợp câu trả lời...
                </div>
              ) : answer ? (
                <>
                  <div className="answer-heading">
                    <Bot size={22} />
                    <div>
                      <p className="eyebrow">Câu hỏi</p>
                      <h3>{answer.question}</h3>
                    </div>
                  </div>
                  <p className="conclusion">{answer.conclusion}</p>
                  <div className="answer-columns">
                    <div>
                      <h4>Điểm tích cực</h4>
                      {answer.positives.length > 0 ? (
                        <ul>
                          {answer.positives.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className="muted">Không đủ bằng chứng tích cực.</p>
                      )}
                    </div>
                    <div>
                      <h4>Lưu ý</h4>
                      <ul>
                        {answer.cautions.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                  <div className="citations">
                    <h4>Dẫn chứng review</h4>
                    {selectedCitationReviews.length > 0 ? (
                      selectedCitationReviews.map((review) => (
                        <button key={review.id} onClick={() => handleCitationClick(review.id)} type="button">
                          {review.id} · {review.rating} sao · rerank {review.rerank.toFixed(2)}
                        </button>
                      ))
                    ) : (
                      <p className="empty-evidence">
                        <ShieldAlert size={16} /> Không tìm thấy đủ bằng chứng trong đánh giá đã thu thập.
                      </p>
                    )}
                  </div>
                </>
              ) : null}
            </section>
          </article>

          <article className="panel debug-panel">
            <div className="panel-title">
              <Layers3 size={18} />
              <h2>RAG Debug View</h2>
            </div>
            <dl>
              <div>
                <dt>Query gốc</dt>
                <dd>{answer?.question ?? question}</dd>
              </div>
              <div>
                <dt>Embedding</dt>
                <dd>{answer?.embeddingModel ?? 'bge-m3'} · vector[1024]</dd>
              </div>
              <div>
                <dt>Retrieval</dt>
                <dd>ChromaDB/FAISS · top-k={filteredReviews.length}</dd>
              </div>
              <div>
                <dt>Rerank</dt>
                <dd>{useRerank ? 'BGE-Reranker enabled' : 'Disabled for comparison'}</dd>
              </div>
              <div>
                <dt>Latency</dt>
                <dd>{answer?.latency ?? 0}ms</dd>
              </div>
            </dl>
            <div className="prompt-preview">
              <h3>Prompt context</h3>
              <pre>{answer?.promptContext ?? 'Chưa có context.'}</pre>
            </div>
          </article>
        </section>
      </section>

      <aside className="evidence-panel">
        <div className="panel-title">
          <Search size={18} />
          <h2>Review Evidence</h2>
        </div>
        <div className="topic-list">
          {summary.topics.map((topic) => (
            <span key={topic.name}>
              {topic.name} <b>{topic.count}</b>
            </span>
          ))}
        </div>
        <div className="pros-cons">
          <div>
            <h3>Ưu điểm</h3>
            {summary.pros.map((item) => (
              <p key={item}>{item}</p>
            ))}
          </div>
          <div>
            <h3>Nhược điểm</h3>
            {summary.cons.map((item) => (
              <p key={item}>{item}</p>
            ))}
          </div>
        </div>
        <div className="review-list">
          {filteredReviews.length > 0 ? (
            filteredReviews.map((review) => (
              <div
                className={`review-card ${activeCitation === review.id ? 'highlight' : ''}`}
                key={review.id}
                ref={(element) => {
                  evidenceRefs.current[review.id] = element
                }}
              >
                <div className="review-meta">
                  <strong>{review.id}</strong>
                  <span>{review.rating} sao</span>
                  <span>{review.source}</span>
                </div>
                <p>{review.text}</p>
                <div className="score-row">
                  <span>similarity {review.similarity.toFixed(2)}</span>
                  <span>rerank {review.rerank.toFixed(2)}</span>
                  <span>{new Date(review.date).toLocaleDateString('vi-VN')}</span>
                </div>
              </div>
            ))
          ) : (
            <p className="empty-evidence">
              <ShieldAlert size={16} /> Không có review phù hợp với bộ lọc hiện tại.
            </p>
          )}
        </div>
      </aside>
    </main>
  )
}

export default App
