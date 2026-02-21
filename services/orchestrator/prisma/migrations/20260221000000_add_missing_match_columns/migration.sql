-- AlterTable: Add missing columns to Match
ALTER TABLE "Match" ADD COLUMN "progress" INTEGER NOT NULL DEFAULT 0;
ALTER TABLE "Match" ADD COLUMN "highlightReelUrl" TEXT;

-- AlterTable: Add missing videoUrl column to Highlight
ALTER TABLE "Highlight" ADD COLUMN "videoUrl" TEXT;

-- CreateIndex
CREATE INDEX IF NOT EXISTS "Event_matchId_idx" ON "Event"("matchId");
CREATE INDEX IF NOT EXISTS "Event_timestamp_idx" ON "Event"("timestamp");
CREATE INDEX IF NOT EXISTS "EmotionScore_matchId_idx" ON "EmotionScore"("matchId");
CREATE INDEX IF NOT EXISTS "EmotionScore_timestamp_idx" ON "EmotionScore"("timestamp");
CREATE INDEX IF NOT EXISTS "Highlight_matchId_idx" ON "Highlight"("matchId");
CREATE INDEX IF NOT EXISTS "Highlight_startTime_idx" ON "Highlight"("startTime");

-- AlterEnum: Add new event types
ALTER TYPE "EventType" ADD VALUE IF NOT EXISTS 'HIGHLIGHT';
ALTER TYPE "EventType" ADD VALUE IF NOT EXISTS 'PENALTY';
ALTER TYPE "EventType" ADD VALUE IF NOT EXISTS 'RED_CARD';
ALTER TYPE "EventType" ADD VALUE IF NOT EXISTS 'YELLOW_CARD';
ALTER TYPE "EventType" ADD VALUE IF NOT EXISTS 'CORNER';
ALTER TYPE "EventType" ADD VALUE IF NOT EXISTS 'OFFSIDE';
ALTER TYPE "EventType" ADD VALUE IF NOT EXISTS 'CELEBRATION';

-- Fix foreign keys to CASCADE delete
ALTER TABLE "Event" DROP CONSTRAINT "Event_matchId_fkey";
ALTER TABLE "Event" ADD CONSTRAINT "Event_matchId_fkey" FOREIGN KEY ("matchId") REFERENCES "Match"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "EmotionScore" DROP CONSTRAINT "EmotionScore_matchId_fkey";
ALTER TABLE "EmotionScore" ADD CONSTRAINT "EmotionScore_matchId_fkey" FOREIGN KEY ("matchId") REFERENCES "Match"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "Highlight" DROP CONSTRAINT "Highlight_matchId_fkey";
ALTER TABLE "Highlight" ADD CONSTRAINT "Highlight_matchId_fkey" FOREIGN KEY ("matchId") REFERENCES "Match"("id") ON DELETE CASCADE ON UPDATE CASCADE;
