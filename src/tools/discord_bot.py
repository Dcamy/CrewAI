# iChain/src/iChain/tools/discord_bot.py

import discord
from discord.ext import commands
import os
import logging

# Setup logging configuration
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'discord_bot_log.txt'), level=logging.INFO)

logger = logging.getLogger(__name__)

class DiscordBotTool:
    def __init__(self, token):
        """
        Initializes the Discord bot with necessary intents and configurations.
        """
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.token = token

        # Register event handlers and commands
        self.bot.event(self.on_ready)
        self.bot.command(name="info", help="Provides information based on the user's query")(self.info)

    async def on_ready(self):
        """
        Event handler for when the bot logs in successfully.
        """
        logger.info(f"[DiscordBotTool] Logged in as {self.bot.user}!")

    async def info(self, ctx, *, query: str):
        """
        A command to process queries using CrewAI's system, returning insights directly.
        """
        try:
            response = f"Received your query: {query}"  # Example interaction, replace with actual CrewAI integration
            await ctx.send(response)
            logger.info(f"[DiscordBotTool] Processed query: {query}")
        except Exception as e:
            logger.error(f"[DiscordBotTool] Error processing query: {e}")
            await ctx.send("Error processing your query.")

    def run(self):
        """
        Runs the Discord bot.
        """
        try:
            self.bot.run(self.token)
        except Exception as e:
            logger.error(f"[DiscordBotTool] Error running bot: {e}")