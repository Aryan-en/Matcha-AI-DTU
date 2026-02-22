/*
  Warnings:

  - The values [Celebrate] on the enum `EventType` will be removed. If these variants are still used in the database, this will fail.

*/
-- AlterEnum
BEGIN;
CREATE TYPE "EventType_new" AS ENUM ('GOAL', 'FOUL', 'TACKLE', 'SAVE', 'CELEBRATION', 'HIGHLIGHT', 'PENALTY', 'RED_CARD', 'YELLOW_CARD', 'CORNER', 'OFFSIDE');
ALTER TABLE "Event" ALTER COLUMN "type" TYPE "EventType_new" USING ("type"::text::"EventType_new");
ALTER TYPE "EventType" RENAME TO "EventType_old";
ALTER TYPE "EventType_new" RENAME TO "EventType";
DROP TYPE "EventType_old";
COMMIT;

-- AlterTable
ALTER TABLE "Match" ADD COLUMN     "heatmapUrl" TEXT,
ADD COLUMN     "topSpeedKmh" DOUBLE PRECISION;
