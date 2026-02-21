import {
  Controller,
  Get,
  Post,
  Delete,
  Param,
  UseInterceptors,
  UploadedFile,
  Body,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { MatchesService } from './matches.service';
import { Match } from '@prisma/client';
import 'multer';

@Controller('matches')
export class MatchesController {
  constructor(private readonly matchesService: MatchesService) {}

  @Post('upload')
  @UseInterceptors(FileInterceptor('file'))
  async uploadFile(@UploadedFile() file: any): Promise<Match> {
    return this.matchesService.create(file);
  }

  @Get()
  async findAll(): Promise<Match[]> {
    return this.matchesService.findAll();
  }

  @Get(':id')
  async findOne(@Param('id') id: string): Promise<Match | null> {
    return this.matchesService.findOne(id);
  }

  @Post(':id/progress')
  async updateProgress(
    @Param('id') id: string,
    @Body() body: { progress: number },
  ) {
    return this.matchesService.updateProgress(id, body.progress);
  }

  @Post(':id/live-event')
  async addLiveEvent(@Param('id') id: string, @Body() body: any) {
    return this.matchesService.addLiveEvent(id, body);
  }

  @Post(':id/complete')
  async completeMatch(@Param('id') id: string, @Body() body: any) {
    return this.matchesService.completeMatch(id, body);
  }

  @Post(':id/reanalyze')
  async reanalyzeMatch(@Param('id') id: string): Promise<{ ok: boolean }> {
    return this.matchesService.reanalyzeMatch(id);
  }

  @Delete(':id')
  async deleteMatch(@Param('id') id: string): Promise<{ ok: boolean }> {
    return this.matchesService.deleteMatch(id);
  }
}
