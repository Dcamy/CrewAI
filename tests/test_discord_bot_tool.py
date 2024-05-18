# iChain/tests/test_discord_bot_tool.py
import unittest
from unittest.mock import MagicMock, patch
from iChain.src.iChain.tools.discord_bot import DiscordBotTool
import discord
import os

class TestDiscordBotTool(unittest.TestCase):
    """
    This class contains tests for the DiscordBotTool.
    """

    @patch('discord.ext.commands.Bot.run')
    @patch('discord.ext.commands.Bot.__init__', return_value=None)
    def setUp(self, mock_bot_init, mock_bot_run):
        """
        Set up environment before each test.
        """
        self.bot_token = "dummy_token"
        self.bot_tool = DiscordBotTool(self.bot_token)

    @patch('discord.ext.commands.Context.send')
    async def test_info_command(self, mock_send):
        """
        Test that the info command processes queries correctly.
        """
        ctx = MagicMock()
        query = "What is the weather today?"

        await self.bot_tool.info(ctx, query=query)
        mock_send.assert_called_once_with(f"Received your query: {query}")

    @patch('discord.ext.commands.Context.send')
    async def test_info_command_with_exception(self, mock_send):
        """
        Test that the info command handles exceptions correctly.
        """
        ctx = MagicMock()
        query = "What is the weather today?"

        with patch.object(self.bot_tool, 'info', side_effect=Exception("Test exception")):
            await self.bot_tool.info(ctx, query=query)
            mock_send.assert_called_once_with("Error processing your query.")

    @patch('discord.ext.commands.Bot.run')
    def test_run(self, mock_run):
        """
        Test that the bot runs with the provided token.
        """
        self.bot_tool.run()
        mock_run.assert_called_once_with(self.bot_token)

    @patch('discord.ext.commands.Bot.run', side_effect=Exception("Run error"))
    def test_run_with_exception(self, mock_run):
        """
        Test that the bot handles exceptions during run.
        """
        with self.assertRaises(Exception):
            self.bot_tool.run()
        mock_run.assert_called_once_with(self.bot_token)

    def test_bot_token_not_set(self):
        """
        Test that a ValueError is raised if the bot token is not set.
        """
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                DiscordBotTool(os.getenv("DISCORD_BOT_TOKEN"))