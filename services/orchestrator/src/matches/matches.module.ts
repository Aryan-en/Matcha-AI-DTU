import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { MatchesService } from './matches.service';
import { MatchesController } from './matches.controller';
import { EventsModule } from '../events/events.module';

@Module({
  imports: [HttpModule, EventsModule],
  controllers: [MatchesController],
  providers: [MatchesService],
})
export class MatchesModule {}
