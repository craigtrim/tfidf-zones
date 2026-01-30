# =============================================================================
# WORD LISTS: FUNCTION WORDS & STOP WORDS
# =============================================================================
#
# Comprehensive, pedantically curated lists of English function words and
# stop words for use in token classification and filtering.
#
# FUNCTION_WORDS: Closed-class grammatical words that serve structural
#   purposes rather than carrying lexical/content meaning.  Subcategorised
#   by traditional part-of-speech role.
#
# STOP_WORDS: High-frequency content-adjacent words that add little
#   discriminative value in information retrieval but are NOT function
#   words.  This set is guaranteed disjoint from FUNCTION_WORDS.
#
# Design note: the project's TF-IDF pipeline intentionally preserves
# function words (they carry stylometric signal and IDF handles ubiquity).
# These lists exist as reference data — they are not applied as filters
# by default.
#
# =============================================================================

from __future__ import annotations

# ── Function Words ──────────────────────────────────────────────────────────
#
# Closed-class words grouped by grammatical category.  Every entry is
# lowercase.  The union of all sub-lists is exported as FUNCTION_WORDS.

# ---- Determiners / Articles ------------------------------------------------
_DETERMINERS: list[str] = [
    # articles
    "a", "an", "the",
    # demonstratives
    "this", "that", "these", "those",
    # distributives
    "each", "every", "either", "neither",
    # quantifiers (determiner use)
    "all", "any", "both", "few", "fewer", "fewest",
    "half", "least", "less", "little",
    "many", "more", "most", "much",
    "no", "none",
    "several", "some",
    "enough", "sufficient",
    # possessive determiners
    "my", "your", "his", "her", "its", "our", "their",
    # interrogative / relative determiners
    "what", "which", "whose",
    # other determiners
    "another", "other", "such",
    "certain",  # determiner use: "certain people"
    "own",      # post-possessive determiner: "my own"
    "said",     # legal/formal determiner: "said property"
    "whatsoever",  # determiner: "no reason whatsoever"
]

# ---- Pronouns --------------------------------------------------------------
_PRONOUNS: list[str] = [
    # personal — subject
    "i", "you", "he", "she", "it", "we", "they",
    # personal — object
    "me", "him", "us", "them",
    # reflexive
    "myself", "yourself", "yourselves",
    "himself", "herself", "itself",
    "ourselves", "themselves",
    "oneself", "themself",  # singular they reflexive
    # possessive (standalone)
    "mine", "yours", "hers", "ours", "theirs",
    # demonstrative (pronoun use)
    "this", "that", "these", "those",
    # indefinite
    "all", "another", "any", "anybody", "anyone", "anything",
    "both", "each", "either", "enough",
    "everybody", "everyone", "everything",
    "few", "fewer",
    "little", "less",
    "many", "more", "most", "much",
    "neither", "nobody", "none", "nothing",
    "one", "ones",
    "other", "others",
    "several", "some", "somebody", "someone", "something",
    "such",
    "naught", "nought",  # archaic/formal: "all for naught"
    "aught",             # archaic: "for aught I know"
    "whatnot",           # indefinite: "books and whatnot"
    "suchlike",          # indefinite: "dogs, cats, and suchlike"
    # interrogative
    "who", "whom", "whose", "what", "which",
    # relative
    "that", "which", "who", "whom", "whose",
    "whoever", "whomever", "whatever", "whichever",
]

# ---- Prepositions -----------------------------------------------------------
_PREPOSITIONS: list[str] = [
    "aboard", "about", "above", "across", "after", "against",
    "along", "alongside", "amid", "amidst", "among", "amongst",
    "around", "as", "astride", "at", "atop",
    "barring", "before", "behind", "below", "beneath", "beside", "besides",
    "between", "beyond",
    "but",  # prepositional use: "everyone but him"
    "by",
    "circa", "concerning",
    "despite", "down", "during",
    "except", "excluding",
    "following", "for", "from",
    "given",
    "in", "including", "inside", "into",
    "less",  # "less tax"
    "like",
    "mid", "midst", "minus",
    "near", "nearer", "nearest", "next",
    "notwithstanding",
    "of", "off", "on", "onto", "opposite", "out", "outside", "over",
    "past", "pending", "per", "plus",
    "regarding", "round",
    "save", "since",
    "than", "through", "throughout", "till", "to", "toward", "towards",
    "under", "underneath", "unlike", "until", "unto", "up", "upon",
    "versus", "via",
    "with", "within", "without",
    "worth",
    # less common / formal / archaic prepositions
    "absent",   # "absent any evidence"
    "anti",     # "anti war"
    "bar",      # "bar none"
    "cum",      # "bedroom-cum-study"
    "ere",      # archaic: "ere long"
    "lest",     # also conjunction, prepositional shade
    "pro",      # "pro independence"
    "qua",      # "art qua art"
    "re",       # "re your letter"
    "sans",     # "sans serif"
    "thru",     # informal variant of "through"
]

# ---- Conjunctions -----------------------------------------------------------
_CONJUNCTIONS: list[str] = [
    # coordinating
    "and", "but", "or", "nor", "for", "yet", "so",
    # subordinating
    "after", "although", "as", "because", "before",
    "even", "if", "lest",
    "once", "only", "provided", "since",
    "than", "that", "though",
    "till", "unless", "until", "when",
    "whenever", "where", "wherever",
    "whereas", "whereby", "whether", "while", "whilst",
    # correlative (individual words)
    "both", "either", "neither", "not", "only", "whether",
    # additional subordinating
    "albeit",     # "albeit slowly"
    "else",       # conjunction: "or else"
    "inasmuch",   # "inasmuch as"
    "insofar",    # "insofar as"
    "insomuch",   # "insomuch that"
    "supposing",  # "supposing that"
    "howbeit",    # archaic: "be that as it may"
    "notwithstanding",  # conjunction use
    "wherefore",  # archaic: "wherefore art thou"
]

# ---- Auxiliary & Modal Verbs ------------------------------------------------
_AUXILIARIES: list[str] = [
    # primary auxiliaries
    "be", "am", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    # modals
    "can", "could",
    "may", "might",
    "shall", "should",
    "will", "would",
    "must",
    "need",  # modal use: "need not"
    "dare",  # modal use: "dare not"
    "ought",
    # semi-modals / quasi-modals
    "used",  # "used to"
    # negative contractions (expanded forms are separate tokens after
    # contraction expansion, but raw text may contain these)
    "cannot",
]

# ---- Adverbs (function-word adverbs only) -----------------------------------
#
# Only closed-class, grammatical adverbs — degree adverbs, conjunctive
# adverbs, focus adverbs, and negation.  Content adverbs (e.g. "quickly",
# "loudly") are NOT included.
_ADVERBS: list[str] = [
    # negation
    "not", "never", "neither", "nor",
    # degree / intensifiers
    "almost", "also", "altogether",
    "awfully",
    "barely",
    "completely",
    "entirely", "especially", "even", "ever",
    "extremely",
    "fairly", "far", "further", "furthermore",
    "hardly",
    "however",
    "indeed",
    "just",
    "merely",
    "nearly",
    "only",
    "otherwise",
    "partly", "perhaps",
    "pretty",  # degree adverb: "pretty good"
    "quite",
    "rather", "really",
    "scarcely", "simply", "slightly", "somewhat",
    "still",
    "terribly", "thoroughly", "thus",
    "too",
    "utterly",
    "very",
    "well",  # degree: "well aware"
    "wholly",
    # conjunctive adverbs
    "accordingly", "additionally",
    "also",
    "besides",
    "consequently", "conversely",
    "hence", "henceforth",
    "instead",
    "likewise",
    "meanwhile", "moreover",
    "nevertheless", "nonetheless",
    "otherwise",
    "similarly",
    "subsequently",
    "then", "thereafter", "thereby", "therefore",
    "thus",
    # temporal function adverbs
    "again", "already", "always",
    "here",
    "hitherto",
    "never", "now",
    "often",
    "once",
    "seldom",
    "sometimes", "soon",
    "then",
    "there",
    "today", "tomorrow",
    "usually",
    "when", "whenever",
    "where", "wherever",
    "yet",
    # interrogative adverbs
    "how", "when", "where", "wherever", "why",
    # relative adverbs
    "when", "where", "why",
    # locative / directional (closed-class or semi-closed)
    "abroad",
    "ahead",
    "anywhere", "anyplace",
    "away",
    "back",
    "downstairs", "downward", "downwards",
    "elsewhere",
    "everywhere",
    "forth",
    "forward", "forwards",
    "hence",
    "hither",   # archaic: "come hither"
    "home",     # adverbial: "go home"
    "indoors",
    "inward", "inwards",
    "nowhere",
    "onward", "onwards",
    "outdoors",
    "outward", "outwards",
    "overhead",
    "sideways",
    "someplace", "somewhere",
    "thence",   # "from thence"
    "thither",  # archaic: "hither and thither"
    "upstairs", "upward", "upwards",
    "whence",   # "from whence"
    "whither",  # archaic: "whither goest thou"
    "backward", "backwards",
    # compound pronominal / locative adverbs
    "heretofore",
    "hereafter", "herein", "hereof", "hereto", "herewith",
    "therein", "thereof", "thereto", "thereupon", "therewith",
    "wherein", "whereof", "whereto", "whereupon",
    # additional temporal
    "afterward", "afterwards",
    "ago",
    "anymore",
    "beforehand",
    "erstwhile",
    "forever", "forevermore",
    "formerly",
    "hereafter",
    "hitherto",
    "nowadays",
    "presently",
    "sometime",
    "straightaway",
    "thenceforth",
    "twice", "thrice",
    "yesterday",
    # additional degree / manner (closed-class)
    "altogether",
    "anyhow", "anyway", "anyways",
    "nowise", "nohow",
    "otherwise",
    "somehow",
    "whatsoever",  # adverb: "none whatsoever"
    "withal",      # archaic: "therewith"
    # additive / restrictive
    "namely",
    "viz",   # abbreviation of videlicet, "namely"
    "alone", # focus adverb: "he alone knew"
    "else",  # adverb: "what else", "someone else"
]

# ---- Particles & Interjections (function-class) -----------------------------
_PARTICLES: list[str] = [
    # infinitive marker
    "to",
    # negative particle
    "not",
    # existential "there"
    "there",
    # discourse particles
    "oh", "ah", "eh", "uh", "um", "hmm",
    "well", "now", "so", "then",
    "please", "yes", "no",
    "okay", "ok",
    # additional interjections / discourse
    "aye", "nay",
    "yep", "yeah", "yea", "yup",
    "nope", "nah",
    "hey", "hi", "hello",
    "huh", "ooh", "oops", "ow", "wow",
    "alas",      # archaic interjection
    "lo",        # archaic: "lo and behold"
    "behold",    # archaic interjection
    "forsooth",  # archaic: "in truth"
    "pray",      # archaic particle: "pray tell"
    "prithee",   # archaic: "I pray thee"
    "indeed",    # also adverb, particle shade
    "quite",     # British discourse particle: "quite"
]

# ---- Expletives & Pro-forms -------------------------------------------------
_EXPLETIVES: list[str] = [
    "it",    # expletive "it": "it is raining"
    "there", # expletive "there": "there is a problem"
]

# ---- Build the unified set --------------------------------------------------

_ALL_FUNCTION_LISTS: list[list[str]] = [
    _DETERMINERS,
    _PRONOUNS,
    _PREPOSITIONS,
    _CONJUNCTIONS,
    _AUXILIARIES,
    _ADVERBS,
    _PARTICLES,
    _EXPLETIVES,
]

FUNCTION_WORDS: frozenset[str] = frozenset(
    word
    for sublist in _ALL_FUNCTION_LISTS
    for word in sublist
)
"""Frozen set of all English function words (closed-class, grammatical)."""


# ── Stop Words ──────────────────────────────────────────────────────────────
#
# High-frequency words commonly removed in IR/NLP pipelines that are NOT
# already in FUNCTION_WORDS.  This includes:
#   - light verbs and copular complements in their content-verb readings
#   - high-frequency content verbs with low discriminative power
#   - common temporal/spatial nouns used adverbially
#   - generic discourse/hedge words
#   - ordinals and quantifier-adjacent words
#   - contractions and variant forms
#
# Every entry is lowercase.  Guaranteed disjoint from FUNCTION_WORDS.

_STOP_WORDS_RAW: list[str] = [
    # ---- Light / high-frequency verbs (content readings) --------------------
    "become", "becomes", "became", "becoming",
    "begin", "begins", "began", "begun", "beginning",
    "bring", "brings", "brought", "bringing",
    "call", "calls", "called", "calling",
    "come", "comes", "came", "coming",
    "contain", "contains", "contained", "containing",
    "continue", "continues", "continued", "continuing",
    "end", "ends", "ended", "ending",
    "find", "finds", "found", "finding",
    "get", "gets", "got", "gotten", "getting",
    "give", "gives", "gave", "given", "giving",
    "go", "goes", "went", "gone", "going",
    "happen", "happens", "happened", "happening",
    "hold", "holds", "held", "holding",
    "include", "includes", "included",
    "involve", "involves", "involved", "involving",
    "keep", "keeps", "kept", "keeping",
    "know", "knows", "knew", "known", "knowing",
    "lead", "leads", "led", "leading",
    "leave", "leaves", "left", "leaving",
    "let", "lets", "letting",
    "look", "looks", "looked", "looking",
    "make", "makes", "made", "making",
    "mean", "means", "meant", "meaning",
    "move", "moves", "moved", "moving",
    "occur", "occurs", "occurred", "occurring",
    "open", "opens", "opened", "opening",
    "part", "parts",
    "pass", "passes", "passed", "passing",
    "place", "places", "placed", "placing",
    "play", "plays", "played", "playing",
    "point", "points", "pointed", "pointing",
    "provide", "provides", "provided", "providing",
    "put", "puts", "putting",
    "reach", "reaches", "reached", "reaching",
    "read", "reads", "reading",
    "remain", "remains", "remained", "remaining",
    "require", "requires", "required", "requiring",
    "result", "results", "resulted", "resulting",
    "run", "runs", "ran", "running",
    "say", "says", "said", "saying",
    "see", "sees", "saw", "seen", "seeing",
    "seem", "seems", "seemed", "seeming",
    "set", "sets", "setting",
    "show", "shows", "showed", "shown", "showing",
    "stand", "stands", "stood", "standing",
    "start", "starts", "started", "starting",
    "state", "states", "stated", "stating",
    "stop", "stops", "stopped", "stopping",
    "suggest", "suggests", "suggested", "suggesting",
    "take", "takes", "took", "taken", "taking",
    "talk", "talks", "talked", "talking",
    "tell", "tells", "told", "telling",
    "tend", "tends", "tended", "tending",
    "think", "thinks", "thought", "thinking",
    "try", "tries", "tried", "trying",
    "turn", "turns", "turned", "turning",
    "use", "uses", "using",
    "want", "wants", "wanted", "wanting",
    "work", "works", "worked", "working",
    "write", "writes", "wrote", "written", "writing",

    # ---- Additional high-frequency verbs ------------------------------------
    "add", "adds", "added", "adding",
    "agree", "agrees", "agreed", "agreeing",
    "allow", "allows", "allowed", "allowing",
    "appear", "appears", "appeared", "appearing",
    "apply", "applies", "applied", "applying",
    "ask", "asks", "asked", "asking",
    "believe", "believes", "believed", "believing",
    "build", "builds", "built", "building",
    "buy", "buys", "bought", "buying",
    "carry", "carries", "carried", "carrying",
    "cause", "causes", "caused", "causing",
    "change", "changes", "changed", "changing",
    "check", "checks", "checked", "checking",
    "choose", "chooses", "chose", "chosen", "choosing",
    "close", "closes", "closed", "closing",
    "consider", "considers", "considered", "considering",
    "cover", "covers", "covered", "covering",
    "create", "creates", "created", "creating",
    "cut", "cuts", "cutting",
    "deal", "deals", "dealt", "dealing",
    "decide", "decides", "decided", "deciding",
    "describe", "describes", "described", "describing",
    "develop", "develops", "developed", "developing",
    "die", "dies", "died", "dying",
    "draw", "draws", "drew", "drawn", "drawing",
    "drive", "drives", "drove", "driven", "driving",
    "drop", "drops", "dropped", "dropping",
    "eat", "eats", "ate", "eaten", "eating",
    "expect", "expects", "expected", "expecting",
    "explain", "explains", "explained", "explaining",
    "face", "faces", "faced", "facing",
    "fall", "falls", "fell", "fallen", "falling",
    "feel", "feels", "felt", "feeling",
    "fight", "fights", "fought", "fighting",
    "fill", "fills", "filled", "filling",
    "follow", "follows", "followed", "following",
    "grow", "grows", "grew", "grown", "growing",
    "hear", "hears", "heard", "hearing",
    "help", "helps", "helped", "helping",
    "hit", "hits", "hitting",
    "hope", "hopes", "hoped", "hoping",
    "imagine", "imagines", "imagined", "imagining",
    "indicate", "indicates", "indicated", "indicating",
    "join", "joins", "joined", "joining",
    "kill", "kills", "killed", "killing",
    "lack", "lacks", "lacked", "lacking",
    "lay", "lays", "laid", "laying",
    "learn", "learns", "learned", "learnt", "learning",
    "lie", "lies", "lay", "lain", "lying",
    "live", "lives", "lived", "living",
    "lose", "loses", "lost", "losing",
    "love", "loves", "loved", "loving",
    "manage", "manages", "managed", "managing",
    "mark", "marks", "marked", "marking",
    "meet", "meets", "met", "meeting",
    "mention", "mentions", "mentioned", "mentioning",
    "mind", "minds", "minded", "minding",
    "miss", "misses", "missed", "missing",
    "note", "notes", "noted", "noting",
    "notice", "notices", "noticed", "noticing",
    "offer", "offers", "offered", "offering",
    "pay", "pays", "paid", "paying",
    "pick", "picks", "picked", "picking",
    "plan", "plans", "planned", "planning",
    "present", "presents", "presented", "presenting",
    "press", "presses", "pressed", "pressing",
    "produce", "produces", "produced", "producing",
    "pull", "pulls", "pulled", "pulling",
    "push", "pushes", "pushed", "pushing",
    "raise", "raises", "raised", "raising",
    "receive", "receives", "received", "receiving",
    "recognize", "recognizes", "recognized", "recognizing",
    "record", "records", "recorded", "recording",
    "reduce", "reduces", "reduced", "reducing",
    "refer", "refers", "referred", "referring",
    "reflect", "reflects", "reflected", "reflecting",
    "relate", "relates", "related", "relating",
    "remember", "remembers", "remembered", "remembering",
    "remove", "removes", "removed", "removing",
    "report", "reports", "reported", "reporting",
    "represent", "represents", "represented", "representing",
    "return", "returns", "returned", "returning",
    "rise", "rises", "rose", "risen", "rising",
    "save", "saves", "saved", "saving",
    "seek", "seeks", "sought", "seeking",
    "sell", "sells", "sold", "selling",
    "send", "sends", "sent", "sending",
    "serve", "serves", "served", "serving",
    "share", "shares", "shared", "sharing",
    "sit", "sits", "sat", "sitting",
    "speak", "speaks", "spoke", "spoken", "speaking",
    "spend", "spends", "spent", "spending",
    "stay", "stays", "stayed", "staying",
    "strike", "strikes", "struck", "stricken", "striking",
    "study", "studies", "studied", "studying",
    "support", "supports", "supported", "supporting",
    "suppose", "supposes", "supposed", "supposing",
    "teach", "teaches", "taught", "teaching",
    "test", "tests", "tested", "testing",
    "touch", "touches", "touched", "touching",
    "train", "trains", "trained", "training",
    "travel", "travels", "traveled", "travelling", "travelling",
    "treat", "treats", "treated", "treating",
    "understand", "understands", "understood", "understanding",
    "visit", "visits", "visited", "visiting",
    "wait", "waits", "waited", "waiting",
    "walk", "walks", "walked", "walking",
    "watch", "watches", "watched", "watching",
    "wear", "wears", "wore", "worn", "wearing",
    "win", "wins", "won", "winning",
    "wish", "wishes", "wished", "wishing",
    "wonder", "wonders", "wondered", "wondering",

    # ---- Generic / discourse nouns ------------------------------------------
    "area", "areas",
    "case", "cases",
    "day", "days",
    "end", "ends",
    "example", "examples",
    "fact", "facts",
    "form", "forms",
    "group", "groups",
    "hand", "hands",
    "issue", "issues",
    "kind", "kinds",
    "level", "levels",
    "life",
    "line", "lines",
    "lot", "lots",
    "man", "men",
    "matter", "matters",
    "member", "members",
    "month", "months",
    "name", "names",
    "number", "numbers",
    "order",
    "part", "parts",
    "people", "person", "persons",
    "place", "places",
    "point", "points",
    "problem", "problems",
    "program", "programs",
    "question", "questions",
    "reason", "reasons",
    "right", "rights",
    "room", "rooms",
    "side", "sides",
    "sort", "sorts",
    "state", "states",
    "story", "stories",
    "thing", "things",
    "time", "times",
    "type", "types",
    "way", "ways",
    "week", "weeks",
    "woman", "women",
    "word", "words",
    "world",
    "year", "years",

    # ---- Additional generic nouns -------------------------------------------
    "age", "ages",
    "amount", "amounts",
    "answer", "answers",
    "attention",
    "basis",
    "bit", "bits",
    "body", "bodies",
    "book", "books",
    "boy", "boys",
    "business",
    "center", "centres",
    "chance", "chances",
    "child", "children",
    "city", "cities",
    "class", "classes",
    "company", "companies",
    "condition", "conditions",
    "control",
    "country", "countries",
    "course",
    "deal", "deals",
    "development",
    "door", "doors",
    "effect", "effects",
    "effort", "efforts",
    "experience",
    "eye", "eyes",
    "face", "faces",
    "family", "families",
    "father", "fathers",
    "field", "fields",
    "figure", "figures",
    "food",
    "force", "forces",
    "friend", "friends",
    "girl", "girls",
    "government",
    "ground",
    "head", "heads",
    "history",
    "home", "homes",
    "hour", "hours",
    "house", "houses",
    "idea", "ideas",
    "information",
    "interest", "interests",
    "job", "jobs",
    "land",
    "law", "laws",
    "letter", "letters",
    "light",
    "market",
    "mind", "minds",
    "minute", "minutes",
    "moment", "moments",
    "money",
    "morning",
    "mother", "mothers",
    "movement",
    "need", "needs",
    "night", "nights",
    "office",
    "others",
    "paper", "papers",
    "party", "parties",
    "period", "periods",
    "picture", "pictures",
    "piece", "pieces",
    "position", "positions",
    "power", "powers",
    "price", "prices",
    "process",
    "product", "products",
    "rate", "rates",
    "report", "reports",
    "rest",
    "role", "roles",
    "school", "schools",
    "sense",
    "series",
    "service", "services",
    "situation", "situations",
    "society",
    "space",
    "step", "steps",
    "street",
    "stuff",
    "subject", "subjects",
    "system", "systems",
    "table", "tables",
    "term", "terms",
    "value", "values",
    "view", "views",
    "voice", "voices",
    "war", "wars",
    "water",

    # ---- Generic adjectives / adjectivals -----------------------------------
    "able",
    "bad", "worse", "worst",
    "big", "bigger", "biggest",
    "clear", "clearer", "clearest",
    "different",
    "early", "earlier", "earliest",
    "easy", "easier", "easiest",
    "free",
    "full",
    "general",
    "good", "better", "best",
    "great", "greater", "greatest",
    "hard", "harder", "hardest",
    "high", "higher", "highest",
    "important",
    "large", "larger", "largest",
    "last",
    "late", "later", "latest",
    "likely",
    "long", "longer", "longest",
    "low", "lower", "lowest",
    "main",
    "major",
    "new", "newer", "newest",
    "next",
    "old", "older", "oldest",
    "particular",
    "possible",
    "public",
    "real",
    "recent",
    "right",
    "same",
    "short", "shorter", "shortest",
    "significant",
    "similar",
    "simple", "simpler", "simplest",
    "small", "smaller", "smallest",
    "special",
    "strong", "stronger", "strongest",
    "sure",
    "true",
    "various",
    "whole",
    "young", "younger", "youngest",

    # ---- Additional generic adjectives --------------------------------------
    "available",
    "basic",
    "broad", "broader", "broadest",
    "close", "closer", "closest",
    "common", "commoner", "commonest",
    "complete",
    "concerned",
    "current",
    "dark", "darker", "darkest",
    "deep", "deeper", "deepest",
    "direct",
    "entire",
    "equal",
    "essential",
    "final",
    "fine", "finer", "finest",
    "former",
    "heavy", "heavier", "heaviest",
    "hot", "hotter", "hottest",
    "human",
    "individual",
    "initial",
    "key",
    "known",
    "little", "littler", "littlest",
    "local",
    "modern",
    "natural",
    "necessary",
    "normal",
    "open",
    "original",
    "past",
    "personal",
    "physical",
    "poor", "poorer", "poorest",
    "present",
    "previous",
    "primary",
    "private",
    "proper",
    "quick", "quicker", "quickest",
    "ready",
    "regular",
    "related",
    "serious",
    "single",
    "social",
    "standard",
    "total",
    "usual",
    "wide", "wider", "widest",
    "wrong",

    # ---- Ordinals and number-adjacent words ---------------------------------
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "hundred", "hundreds",
    "thousand", "thousands",
    "million", "millions",
    "billion", "billions",
    "dozen", "dozens",
    "couple",
    "single",
    "double", "triple",

    # ---- Hedge / discourse markers ------------------------------------------
    "actually",
    "apparently",
    "basically",
    "certainly",
    "clearly",
    "definitely",
    "effectively",
    "essentially",
    "evidently",
    "exactly",
    "generally",
    "hopefully",
    "ideally",
    "importantly",
    "incidentally",
    "inevitably",
    "interestingly",
    "largely",
    "mainly",
    "mostly",
    "naturally",
    "necessarily",
    "normally",
    "notably",
    "obviously",
    "occasionally",
    "originally",
    "overall",
    "particularly",
    "personally",
    "possibly",
    "potentially",
    "practically",
    "precisely",
    "presumably",
    "previously",
    "primarily",
    "probably",
    "properly",
    "purely",
    "rarely",
    "readily",
    "recently",
    "relatively",
    "reportedly",
    "respectively",
    "roughly",
    "seriously",
    "significantly",
    "specifically",
    "strictly",
    "strongly",
    "supposedly",
    "surely",
    "technically",
    "typically",
    "ultimately",
    "undoubtedly",
    "unfortunately",
    "usually",
    "virtually",

    # ---- Contractions (as single tokens) ------------------------------------
    "ain't",
    "aren't",
    "can't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "hadn't",
    "hasn't",
    "haven't",
    "he'd", "he'll", "he's",
    "here's",
    "how's",
    "i'd", "i'll", "i'm", "i've",
    "isn't",
    "it'd", "it'll", "it's",
    "let's",
    "mightn't",
    "mustn't",
    "needn't",
    "oughtn't",
    "shan't",
    "she'd", "she'll", "she's",
    "shouldn't",
    "somebody's", "someone's",
    "that's",
    "there'd", "there'll", "there's",
    "they'd", "they'll", "they're", "they've",
    "wasn't",
    "we'd", "we'll", "we're", "we've",
    "weren't",
    "what'll", "what's", "what've",
    "when's",
    "where's",
    "who'd", "who'll", "who's", "who've",
    "why's",
    "won't",
    "wouldn't",
    "you'd", "you'll", "you're", "you've",

    # ---- Additional hedge / discourse markers --------------------------------
    "absolutely",
    "accordingly",
    "admittedly",
    "altogether",
    "broadly",
    "commonly",
    "comparatively",
    "considerably",
    "continually",
    "conversely",
    "correspondingly",
    "curiously",
    "directly",
    "distinctly",
    "doubtless",
    "equally",
    "explicitly",
    "extraordinarily",
    "extremely",
    "fairly",
    "finally",
    "firstly",
    "formally",
    "frequently",
    "fundamentally",
    "gradually",
    "immediately",
    "implicitly",
    "increasingly",
    "independently",
    "indirectly",
    "inherently",
    "initially",
    "instantly",
    "invariably",
    "ironically",
    "justly",
    "lastly",
    "literally",
    "manifestly",
    "markedly",
    "materially",
    "moderately",
    "momentarily",
    "noticeably",
    "objectively",
    "ordinarily",
    "ostensibly",
    "outright",
    "overwhelmingly",
    "plainly",
    "plausibly",
    "predictably",
    "predominantly",
    "presently",
    "progressively",
    "prominently",
    "proportionally",
    "purportedly",
    "reasonably",
    "remarkably",
    "repeatedly",
    "rightly",
    "routinely",
    "secondly",
    "seemingly",
    "simultaneously",
    "solely",
    "steadily",
    "subjectively",
    "substantially",
    "sufficiently",
    "thirdly",
    "traditionally",
    "truly",
    "unanimously",
    "understandably",
    "undeniably",
    "uniformly",
    "universally",
    "unmistakably",
    "variously",
    "wholly",

    # ---- Abbreviation-like / filler tokens ----------------------------------
    "etc",
    "eg",
    "ie",
    "vs",
    "cf",   # "compare"
    "nb",   # "nota bene"
]

# Remove any word that already lives in FUNCTION_WORDS
STOP_WORDS: frozenset[str] = frozenset(
    word for word in _STOP_WORDS_RAW if word not in FUNCTION_WORDS
)
"""Frozen set of English stop words, guaranteed disjoint from FUNCTION_WORDS."""

# Combined set for n-gram filtering
NGRAM_STOP_WORDS: frozenset[str] = FUNCTION_WORDS | STOP_WORDS
"""Union of function words and stop words, used by --no-ngram-stopwords."""


def filter_ngrams(ngrams: list[str], stop_words: frozenset[str] = NGRAM_STOP_WORDS) -> list[str]:
    """Remove n-grams where any constituent token is a stop/function word.

    N-grams use underscore as separator (e.g. "of_the", "in_a").
    """
    return [ng for ng in ngrams if not any(t in stop_words for t in ng.split("_"))]
