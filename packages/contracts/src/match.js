import { z } from "zod";
export const CreateMatchSchema = z.object({
    date: z.string().datetime("Must be a valid ISO Date string"),
    duration: z.number().positive("Duration must be a positive number"),
    location: z.string().min(3, "Location must be at least 3 characters"),
    title: z.string().min(3, "Title must be at least 3 characters"),
});
