import { Injectable, Logger } from '@nestjs/common';
import { PrismaClient, Match, EventType } from '@prisma/client';
import { HttpService } from '@nestjs/axios';
import { EventsGateway } from '../events/events.gateway';
import { firstValueFrom } from 'rxjs';
import * as fs from 'fs';
import * as path from 'path';
import 'multer';

interface IncomingEvent {
  timestamp: number;
  type: string;
  confidence: number;
  finalScore?: number;
  commentary?: string;
}

interface IncomingHighlight {
  startTime: number;
  endTime: number;
  score: number;
  eventType?: string;
  commentary?: string;
  videoUrl?: string;
}

interface IncomingEmotionScore {
  timestamp: number;
  audioScore: number;
  motionScore: number;
  contextWeight: number;
  finalScore: number;
}

interface CompletePayload {
  events: IncomingEvent[];
  highlights?: IncomingHighlight[];
  emotionScores?: IncomingEmotionScore[];
  duration?: number;
  summary?: string;
  highlightReelUrl?: string;
  trackingData?: any[]; // normalised bbox frames [{t, b, p}]
  teamColors?: number[][]; // [[R,G,B],[R,G,B]]
}

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

  async create(file: any): Promise<Match> {
    const fileName = `${Date.now()}-${file.originalname}`;
    const uploadsDir = path.join(process.cwd(), '..', '..', 'uploads');

    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }

    const filePath = path.join(uploadsDir, fileName);
    fs.writeFileSync(filePath, file.buffer);

    // Use the actual filesystem path (works for native Windows execution)
    const publicUrl = `http://localhost:4000/uploads/${fileName}`; // Mock public URL

    const match = await this.prisma.match.create({
      data: {
        uploadUrl: publicUrl,
        status: 'UPLOADED',
        duration: 0,
      },
    });

    this.triggerInference(match.id, filePath);

    return match;
  }

  async triggerInference(matchId: string, videoUrl: string) {
    const inferenceUrl = process.env.INFERENCE_URL || 'http://localhost:8000';
    try {
      this.logger.log(
        `Triggering inference for match ${matchId} at ${videoUrl}`,
      );
      await firstValueFrom(
        this.httpService.post(`${inferenceUrl}/api/v1/analyze`, {
          match_id: matchId,
          video_url: videoUrl, // This needs to be a real path/URL accessible to Docker
        }) as any,
      );
    } catch (error) {
      this.logger.error(`Failed to trigger inference: ${error.message}`);
    }
  }

  async findAll() {
    const matches = await this.prisma.match.findMany({
      orderBy: { createdAt: 'desc' },
      include: {
        _count: { select: { events: true, highlights: true } },
      },
    });
    return matches;
  }

  async findOne(id: string) {
    return this.prisma.match.findUnique({
      where: { id },
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
    this.eventsGateway.server.to(id).emit('matchEvent', {
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
      .catch(() => {});
    this.eventsGateway.server.to(id).emit('progress', { matchId: id, progress });
  }

  async completeMatch(id: string, payload: CompletePayload) {
    const {
      events = [],
      highlights = [],
      emotionScores = [],
      duration,
      summary,
      highlightReelUrl,
      trackingData,
      teamColors,
    } = payload;

    const typeMap: Record<string, EventType> = {
      GOAL: EventType.GOAL,
      FOUL: EventType.FOUL,
      TACKLE: EventType.TACKLE,
      SAVE: EventType.SAVE,
      Celebrate: EventType.Celebrate,
    };
    const validTypes = new Set<string>(Object.values(EventType));

    const validEvents = events
      .map((e) => ({
        matchId: id,
        timestamp: e.timestamp,
        type: typeMap[e.type] ?? EventType.TACKLE,
        confidence: e.confidence,
        finalScore: e.finalScore ?? 0,
        commentary: e.commentary ?? null,
      }))
      .filter((e) => validTypes.has(e.type));

    const validHighlights = (highlights ?? []).map((h) => ({
      matchId: id,
      startTime: h.startTime,
      endTime: h.endTime,
      score: h.score,
      eventType: h.eventType ?? null,
      commentary: h.commentary ?? null,
      videoUrl: h.videoUrl ?? null,
    }));

    const validEmotion = (emotionScores ?? []).map((s) => ({
      matchId: id,
      timestamp: s.timestamp,
      audioScore: s.audioScore,
      motionScore: s.motionScore,
      contextWeight: s.contextWeight,
      finalScore: s.finalScore,
    }));

    await this.prisma.$transaction([
      this.prisma.event.createMany({ data: validEvents }),
      this.prisma.highlight.createMany({ data: validHighlights }),
      this.prisma.emotionScore.createMany({ data: validEmotion }),
      this.prisma.match.update({
        where: { id },
        data: {
          status: 'COMPLETED',
          duration: duration ?? null,
          summary: summary ?? null,
          highlightReelUrl: highlightReelUrl ?? null,
          trackingData: trackingData ?? null,
          teamColors: teamColors ?? null,
        } as any,
      }),
    ]);

    this.eventsGateway.server.to(id).emit('progress', { matchId: id, progress: 100 });
    this.eventsGateway.server.to(id).emit('complete', {
      matchId: id,
      eventCount: validEvents.length,
      highlightCount: validHighlights.length,
    });

    this.logger.log(
      `Match ${id} completed — ${validEvents.length} events, ${validHighlights.length} highlights.`,
    );
    return { ok: true };
  }

  async reanalyzeMatch(id: string): Promise<{ ok: boolean }> {
    const match = await this.prisma.match.findUnique({ where: { id } });
    if (!match) return { ok: false };

    // Wipe previous analysis results, keep the video file
    await this.prisma.$transaction([
      this.prisma.event.deleteMany({ where: { matchId: id } }),
      this.prisma.highlight.deleteMany({ where: { matchId: id } }),
      this.prisma.emotionScore.deleteMany({ where: { matchId: id } }),
      this.prisma.match.update({
        where: { id },
        data: {
          status: 'PROCESSING',
          trackingData: null,
          teamColors: null,
          summary: null,
          duration: null,
        } as any,
      }),
    ]);

    // Reconstruct the filesystem path from the stored public URL
    // uploadUrl format: http://localhost:4000/uploads/<filename>
    const fileName = match.uploadUrl.split('/uploads/').pop() || '';
    const uploadsDir = path.join(process.cwd(), '..', '..', 'uploads');
    const videoPath = path.join(uploadsDir, fileName);

    this.logger.log(`Re-analyzing match ${id} from ${videoPath}`);
    this.triggerInference(id, videoPath);
    return { ok: true };
  }

  async deleteMatch(id: string): Promise<{ ok: boolean }> {
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
