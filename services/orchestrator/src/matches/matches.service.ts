import { Injectable, Logger } from '@nestjs/common';
import { PrismaClient, Match, EventType, Prisma } from '@prisma/client';
import { HttpService } from '@nestjs/axios';
import { EventsGateway } from '../events/events.gateway';
import { firstValueFrom } from 'rxjs';
import * as fs from 'fs';
import * as path from 'path';
import 'multer';
import { WsEvents, isYoutubeUrl, AnalysisPayload } from '@matcha/shared';


@Injectable()
export class MatchesService {
  private prisma: PrismaClient;
  private readonly logger = new Logger(MatchesService.name);

  constructor(
    private eventsGateway: EventsGateway,
    private httpService: HttpService,
  ) {
    this.prisma = new PrismaClient();
  }

  async create(file: Express.Multer.File, userId: string): Promise<Match> {
    if (!file || !file.originalname) {
      throw new Error("Invalid file upload");
    }
    if (file.size > 5000000000) {
      throw new Error("File exceeds 5GB limit");
    }
    
    let filePath: string;
    let fileName: string;

    if (file.path) {
      // Multer diskStorage saved the file
      filePath = file.path;
      fileName = file.filename;
    } else if (file.buffer) {
      // Fallback for memoryStorage
      fileName = `${Date.now()}-${file.originalname}`;
      const uploadsDir = path.join(process.cwd(), '..', '..', 'uploads');
      
      try {
        if (!fs.existsSync(uploadsDir)) {
          fs.mkdirSync(uploadsDir, { recursive: true });
        }
        filePath = path.join(uploadsDir, fileName);
        fs.writeFileSync(filePath, file.buffer);
      } catch (error) {
        this.logger.error(`Failed to save upload: ${(error as Error).message}`);
        throw error;
      }
    } else {
      throw new Error("Invalid file upload: neither buffer nor path found");
    }

    // Use the actual filesystem path (works for native Windows execution)
    const publicUrl = `http://localhost:4000/uploads/${fileName}`; // Mock public URL

    const match = await this.prisma.match.create({
      data: {
        uploadUrl: publicUrl,
        status: 'UPLOADED',
        duration: 0,
        userId,
      },
    });

    this.triggerInference(match.id, filePath);

    return match;
  }

  async createFromYoutube(url: string, userId: string): Promise<Match> {
    if (!url || !isYoutubeUrl(url)) {
      throw new Error("Invalid YouTube URL");
    }

    const match = await this.prisma.match.create({
      data: {
        uploadUrl: url,
        status: 'UPLOADED',
        duration: 0,
        userId,
      },
    });

    this.logger.log(`Created match ${match.id} from YouTube URL: ${url}`);
    
    // Pass the raw YouTube URL to inference.
    // The inference service's analysis.py will intercept it and download via yt-dlp
    this.triggerInference(match.id, url);

    return match;
  }

  async triggerInference(matchId: string, videoUrl: string) {
    const inferenceUrl = process.env.INFERENCE_URL || 'http://localhost:8000';
    const maxAttempts = 5;
    const baseDelayMs = 2000;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        this.logger.log(`Triggering inference for match ${matchId} (attempt ${attempt}/${maxAttempts})`);
        await firstValueFrom(
          this.httpService.post(
            `${inferenceUrl}/api/v1/analyze`,
            { match_id: matchId, video_url: videoUrl },
            { timeout: 10000 },
          ) as any,
        );
        this.logger.log(`Inference triggered successfully for match ${matchId}`);
        return; // Success
      } catch (error) {
        const isLast = attempt === maxAttempts;
        this.logger.error(
          `Inference trigger attempt ${attempt}/${maxAttempts} failed: ${error.message}`,
        );
        if (isLast) {
          // Mark match as FAILED so it's visible in UI
          await this.prisma.match
            .update({ where: { id: matchId }, data: { status: 'FAILED' } })
            .catch(() => {});
          this.eventsGateway.server
            .to(matchId)
            .emit(WsEvents.PROGRESS, { matchId, progress: -1 });
          return;
        }
        // Exponential backoff: 2s, 4s, 8s, 16s
        const delay = baseDelayMs * Math.pow(2, attempt - 1);
        this.logger.log(`Retrying in ${delay}ms...`);
      }
    }
  }

  async findAll(userId?: string) {
    const where = userId 
      ? { OR: [{ userId }, { userId: null }] }
      : { userId: null };

    const matches = await this.prisma.match.findMany({
      where,
      orderBy: { createdAt: 'desc' },
      include: {
        _count: { select: { events: true, highlights: true } },
      },
    });
    return matches;
  }

  async findOne(id: string, userId?: string) {
    const where = userId
      ? { id, OR: [{ userId }, { userId: null }] }
      : { id, userId: null };

    return this.prisma.match.findFirst({
      where,
      include: {
        events: { orderBy: { timestamp: 'asc' } },
        highlights: { orderBy: { startTime: 'asc' } },
        emotionScores: { orderBy: { timestamp: 'asc' } },
      },
    });
  }

  async addLiveEvent(id: string, event: any) {
    /**
     * Called by the inference service for EACH detected event immediately,
     * before the full analysis completes.  We broadcast it via WebSocket so
     * the browser can populate the event feed in real-time.
     * We do NOT save to DB here – the final complete() call saves everything.
     */
    this.eventsGateway.server.to(id).emit(WsEvents.MATCH_EVENT, {
      matchId: id,
      event,
    });
    return { ok: true };
  }

  async updateProgress(id: string, progress: number) {
    // Mark as PROCESSING and save progress to DB
    await this.prisma.match
      .update({
        where: { id },
        data: { status: 'PROCESSING', progress: Math.round(progress) },
      })
      .catch((err: any) => {
        this.logger.warn(`Failed to update progress for match ${id}: ${err.message}`);
      });
    this.eventsGateway.server.to(id).emit(WsEvents.PROGRESS, { matchId: id, progress });
  }

  async completeMatch(id: string, payload: AnalysisPayload) {
    if (!id || !payload) {
      throw new Error("Invalid match ID or payload");
    }
    
    const {
      events = [],
      highlights = [],
      emotionScores = [],
      duration,
      summary,
      highlightReelUrl,
      trackingData,
      teamColors,
      heatmapUrl,
      topSpeedKmh,
      videoUrl,
    } = payload;

    const typeMap: Record<string, EventType> = {
      GOAL: EventType.GOAL,
      FOUL: EventType.FOUL,
      TACKLE: EventType.TACKLE,
      SAVE: EventType.SAVE,
      CELEBRATION: EventType.CELEBRATION,
      Celebrate: EventType.CELEBRATION, // Legacy fallback for old code
      HIGHLIGHT: EventType.HIGHLIGHT,
      PENALTY: EventType.PENALTY,
      RED_CARD: EventType.RED_CARD,
      YELLOW_CARD: EventType.YELLOW_CARD,
      CORNER: EventType.CORNER,
      OFFSIDE: EventType.OFFSIDE,
    };
    const validTypes = new Set<string>(Object.values(EventType));

    const validEvents = events
      .map((e) => ({
        matchId: id,
        timestamp: e.timestamp,
        type: typeMap[e.type] ?? EventType.HIGHLIGHT, // Safe fallback to generic highlight
        confidence: Math.max(0, Math.min(1, e.confidence)), // Clamp 0-1
        finalScore: Math.max(0, Math.min(10, e.finalScore ?? 0)), // Clamp 0-10
        commentary: (e.commentary ?? "").substring(0, 1000), // Cap at 1000 chars
      }))
      .filter((e) => validTypes.has(e.type));

    const validHighlights = (highlights ?? []).map((h) => ({
      matchId: id,
      startTime: h.startTime,
      endTime: h.endTime,
      score: Math.max(0, Math.min(10, h.score)), // Clamp 0-10
      eventType: (h.eventType ?? "").substring(0, 50), // Cap at 50 chars
      commentary: (h.commentary ?? "").substring(0, 500), // Cap at 500 chars
      videoUrl: h.videoUrl ?? null,
    }));

    const validEmotion = (emotionScores ?? []).map((s) => ({
      matchId: id,
      timestamp: s.timestamp,
      audioScore: Math.max(0, Math.min(1, s.audioScore)), // Clamp 0-1
      motionScore: Math.max(0, Math.min(1, s.motionScore)), // Clamp 0-1
      contextWeight: Math.max(0, Math.min(1, s.contextWeight)), // Clamp 0-1
      finalScore: Math.max(0, Math.min(10, s.finalScore)), // Clamp 0-10
    }));

    await this.prisma.$transaction([
      this.prisma.event.createMany({ data: validEvents }),
      this.prisma.highlight.createMany({ data: validHighlights }),
      this.prisma.emotionScore.createMany({ data: validEmotion }),
      this.prisma.match.update({
        where: { id },
        data: {
          status: 'COMPLETED',
          duration: Math.max(0, duration ?? 0), // Ensure non-negative
          summary: (summary ?? '').substring(0, 5000), // Cap at 5000 chars
          highlightReelUrl,
          trackingData,
          teamColors,
          heatmapUrl: heatmapUrl ?? null,
          topSpeedKmh: topSpeedKmh ?? null,
          ...(videoUrl ? { uploadUrl: videoUrl } : {}),
        } as any,
      }),
    ]);

    this.eventsGateway.server.to(id).emit(WsEvents.PROGRESS, { matchId: id, progress: 100 });
    this.eventsGateway.server.to(id).emit(WsEvents.COMPLETE, {
      matchId: id,
      eventCount: validEvents.length,
      highlightCount: validHighlights.length,
    });

    this.logger.log(
      `Match ${id} completed — ${validEvents.length} events, ${validHighlights.length} highlights.`,
    );
    return { ok: true };
  }

  async reanalyzeMatch(id: string, userId: string): Promise<{ ok: boolean }> {
    if (!id || typeof id !== 'string') {
      this.logger.error("Invalid match ID for reanalysis");
      return { ok: false };
    }
    
    const match = await this.prisma.match.findFirst({ where: { id, OR: [{ userId }, { userId: null }] } });
    if (!match) {
      this.logger.warn(`Match not found: ${id}`);
      return { ok: false };
    }

    // Wipe previous analysis results, keep the video file
    await this.prisma.$transaction([
      this.prisma.event.deleteMany({ where: { matchId: id } }),
      this.prisma.highlight.deleteMany({ where: { matchId: id } }),
      this.prisma.emotionScore.deleteMany({ where: { matchId: id } }),
      this.prisma.match.update({
        where: { id },
        data: {
          status: 'PROCESSING',
          progress: 0,
          trackingData: undefined,
          teamColors: undefined,
          heatmapUrl: null,
          topSpeedKmh: null,
          summary: null,
          duration: null,
        } as any,
      }),
    ]);

    // Reconstruct the filesystem path from the stored public URL
    // uploadUrl format: http://localhost:4000/uploads/<filename>
    const uploadsDir = path.join(process.cwd(), '..', '..', 'uploads');
    const uploadUrlParts = match.uploadUrl.split('/uploads/');
    if (uploadUrlParts.length < 2) {
      this.logger.error(`Invalid uploadUrl format: ${match.uploadUrl}`);
      throw new Error('Invalid uploadUrl format');
    }
    const fileName = uploadUrlParts[uploadUrlParts.length - 1];
    const videoPath = path.join(uploadsDir, fileName);

    this.logger.log(`Re-analyzing match ${id} from ${videoPath}`);
    this.triggerInference(id, videoPath);
    return { ok: true };
  }

  async deleteMatch(id: string, userId: string): Promise<{ ok: boolean }> {
    const match = await this.prisma.match.findFirst({ where: { id, OR: [{ userId }, { userId: null }] } });
    if (!match) return { ok: false };

    await this.prisma.$transaction([
      this.prisma.event.deleteMany({ where: { matchId: id } }),
      this.prisma.highlight.deleteMany({ where: { matchId: id } }),
      this.prisma.emotionScore.deleteMany({ where: { matchId: id } }),
      this.prisma.match.delete({ where: { id } }),
    ]);
    this.logger.log(`Match ${id} deleted.`);
    return { ok: true };
  }
}
