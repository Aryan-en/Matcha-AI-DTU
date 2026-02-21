-- AlterTable: change duration from Int to Float, add commentary/finalScore to Event
ALTER TABLE "Event" ADD COLUMN IF NOT EXISTS "finalScore" DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE "Event" ADD COLUMN IF NOT EXISTS "commentary" TEXT;

-- AlterTable: change duration type on Match
ALTER TABLE "Match" ALTER COLUMN "duration" TYPE DOUBLE PRECISION;

-- CreateTable: EmotionScore
CREATE TABLE IF NOT EXISTS "EmotionScore" (
    "id" TEXT NOT NULL,
    "matchId" TEXT NOT NULL,
    "timestamp" DOUBLE PRECISION NOT NULL,
    "audioScore" DOUBLE PRECISION NOT NULL,
    "motionScore" DOUBLE PRECISION NOT NULL,
    "contextWeight" DOUBLE PRECISION NOT NULL,
    "finalScore" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "EmotionScore_pkey" PRIMARY KEY ("id")
);

-- CreateTable: Highlight
CREATE TABLE IF NOT EXISTS "Highlight" (
    "id" TEXT NOT NULL,
    "matchId" TEXT NOT NULL,
    "startTime" DOUBLE PRECISION NOT NULL,
    "endTime" DOUBLE PRECISION NOT NULL,
    "score" DOUBLE PRECISION NOT NULL,
    "eventType" TEXT,
    "commentary" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Highlight_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "EmotionScore" ADD CONSTRAINT "EmotionScore_matchId_fkey"
    FOREIGN KEY ("matchId") REFERENCES "Match"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Highlight" ADD CONSTRAINT "Highlight_matchId_fkey"
    FOREIGN KEY ("matchId") REFERENCES "Match"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
