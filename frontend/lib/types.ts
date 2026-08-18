export type JobStatus = "pending" | "running" | "succeeded" | "failed" | "cancelled";

export interface Job {
  id: string;
  kind: string;
  target: string;
  status: JobStatus;
  progress: number;
  message: string;
  result: IndexReport | null;
  error: string;
  createdAt: number;
  finishedAt: number | null;
  elapsedSeconds: number;
}

export interface IndexReport {
  root: string;
  rootId: string;
  totalFiles: number;
  indexedFiles: number;
  skippedUnchanged: number;
  removedFiles: number;
  chunksWritten: number;
  foldersSummarized: number;
  directories: number;
  totalSizeBytes: number;
  durationSeconds: number;
  truncated: boolean;
  warnings: string[];
}

export interface Health {
  status: "ok" | "degraded";
  ollama: {
    reachable: boolean;
    baseUrl: string;
    models: string[];
    chatModelReady: boolean;
    embeddingModelReady: boolean;
    detail: string;
  };
  indexedRoots: number;
  chatModel: string;
  embeddingModel: string;
  dataDir: string;
}

export interface FileEntry {
  name: string;
  relPath: string;
  ext: string;
  size: number;
  modifiedAt: number;
}

export interface TreeNode {
  name: string;
  relPath: string;
  absPath: string;
  depth: number;
  directFiles: number;
  totalFiles: number;
  totalSize: number;
  summary: string | null;
  truncated: boolean;
  children: TreeNode[];
  files: FileEntry[];
}

export interface TreeResponse {
  root: string;
  totalFiles: number;
  totalSize: number;
  directories: number;
  indexedChunks: number;
  truncated: boolean;
  tree: TreeNode;
}

export interface SearchPlan {
  semanticQuery: string;
  extensions: string[];
  nameContains: string;
  daysBack: number;
  target: "files" | "folders";
}

export interface Source {
  index: number;
  relPath: string;
  absPath: string;
  name: string;
  ext: string;
  kind: string;
  score: number;
  modifiedAt: number | null;
  size: number | null;
  excerpt: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  plan?: SearchPlan;
  streaming?: boolean;
  error?: string;
}

export interface DuplicateGroup {
  digest: string;
  size_bytes: number;
  wasted_bytes: number;
  paths: string[];
  names: string[];
}

export interface Finding {
  rel_path: string;
  reason: string;
  size_bytes: number;
  modified_at: number;
}

export interface CleanupReport {
  root: string;
  filesExamined: number;
  totalSizeBytes: number;
  reclaimableBytes: number;
  reclaimableHuman: string;
  duplicateGroups: DuplicateGroup[];
  nearDuplicates: Finding[];
  emptyFiles: Finding[];
  junkFiles: Finding[];
  staleFiles: Finding[];
  largeFiles: Finding[];
  emptyDirs: string[];
  narrative: string;
}

export interface FilingCandidate {
  relPath: string;
  absPath: string;
  score: number;
  semanticScore: number;
  matches: number;
  summary: string;
  exampleFiles: string[];
}

export interface FilingProposal {
  sourcePath: string;
  destinationDir: string;
  destinationRel: string;
  suggestedFilename: string;
  confidence: number;
  reason: string;
  applied: boolean;
  finalPath: string;
  candidates: FilingCandidate[];
  neighbours: {
    relPath: string;
    name: string;
    score: number;
    excerpt: string;
  }[];
  error: string;
}

export interface RootSummary {
  path: string;
  updatedAt: number;
  files: number;
  chunks: number;
  totalSize: number;
  exists: boolean;
}

export interface WatchStatus {
  root: string;
  rootId?: string;
  watching: boolean;
  reindexing?: boolean;
  pendingChanges?: boolean;
  lastEventAt?: number | null;
  lastIndexedAt?: number | null;
  lastJobId?: string | null;
  lastError?: string;
}
