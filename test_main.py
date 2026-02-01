import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import json
import os
import numpy as np
import main

class TestAllergenius(unittest.TestCase):

    # --- parse_file tests ---
    @patch("builtins.open", new_callable=mock_open, read_data='{"data": [{"QUID": "ingredient 1"}, {"QUID": "ingredient 2"}]}')
    def test_parse_file_success(self, mock_file):
        result = main.parse_file("dummy.json")
        self.assertEqual(result, ["ingredient 1", "ingredient 2"])

    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("builtins.print")
    def test_parse_file_not_found(self, mock_print, mock_file):
        result = main.parse_file("nonexistent.json")
        self.assertEqual(result, [])
        mock_print.assert_called_with("Error: File nonexistent.json not found")

    @patch("builtins.open", side_effect=json.JSONDecodeError("msg", "doc", 0))
    @patch("builtins.print")
    def test_parse_file_invalid_json(self, mock_print, mock_file):
        result = main.parse_file("bad.json")
        self.assertEqual(result, [])
        mock_print.assert_called_with("Error: Invalid JSON format in bad.json")

    # --- save_embeddings tests ---
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_embeddings(self, mock_json_dump, mock_file, mock_exists, mock_makedirs):
        embeddings = [[0.1, 0.2]]
        indices = [0]
        main.save_embeddings("test_emb.json", embeddings, indices)
        
        mock_makedirs.assert_called_with("embeddings")
        mock_file.assert_called_with("embeddings/test_emb.json", "w")
        mock_json_dump.assert_called_with({
            "embeddings": embeddings,
            "indices": indices
        }, mock_file())

    # --- load_embeddings tests ---
    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='{"embeddings": [[0.1]], "indices": [0]}')
    def test_load_embeddings_success(self, mock_file, mock_exists):
        emb, idx = main.load_embeddings("test_emb.json")
        self.assertEqual(emb, [[0.1]])
        self.assertEqual(idx, [0])

    @patch("os.path.exists", return_value=False)
    def test_load_embeddings_not_found(self, mock_exists):
        result = main.load_embeddings("test_emb.json")
        self.assertFalse(result)

    # --- get_embeddings tests ---
    @patch("main.load_embeddings")
    @patch("main.save_embeddings")
    @patch("ollama.embeddings")
    @patch("main.tqdm")
    def test_get_embeddings_new_chunks(self, mock_tqdm, mock_ollama, mock_save, mock_load):
        # Setup: load_embeddings returns empty, so we process chunks
        mock_load.return_value = ([], [])
        # Mock ollama response
        mock_ollama.return_value = {"embedding": [0.1, 0.2]}
        
        chunks = ["chunk1", "chunk2"]
        final_embeddings = main.get_embeddings("test.json", "model", chunks)
        
        self.assertEqual(len(final_embeddings), 2)
        self.assertEqual(mock_ollama.call_count, 2)
        mock_save.assert_called()

    @patch("main.load_embeddings")
    @patch("ollama.embeddings")
    def test_get_embeddings_cached(self, mock_ollama, mock_load):
        # Setup: chunks 0 and 1 are already cached
        mock_load.return_value = ([[0.1], [0.2]], [0, 1])
        
        chunks = ["chunk1", "chunk2"]
        final_embeddings = main.get_embeddings("test.json", "model", chunks)
        
        # Should not call ollama
        mock_ollama.assert_not_called()
        self.assertEqual(len(final_embeddings), 2)

    # --- find_most_similar tests ---
    def test_find_most_similar_success(self):
        needle = [1.0, 0.0]
        haystack = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        # Expected: 1st vector is most similar (dot=1), 2nd is least (dot=0)
        
        results = main.find_most_similar(needle, haystack, top_k=2)
        
        self.assertEqual(len(results), 2)
        # First result should be index 0 with score close to 1.0
        self.assertEqual(results[0][1], 0)
        self.assertAlmostEqual(results[0][0], 1.0, places=5)

    def test_find_most_similar_empty(self):
        self.assertEqual(main.find_most_similar([], []), [])
        self.assertEqual(main.find_most_similar([1], []), [])

    def test_find_most_similar_dimension_mismatch(self):
        needle = [1, 2]
        haystack = [[1, 2, 3]] # Mismatch
        
        # Should print error and return empty list
        with patch("builtins.print") as mock_print:
            result = main.find_most_similar(needle, haystack)
            self.assertEqual(result, [])
            mock_print.assert_any_call("No vectors with matching dimension 2 found")

    # --- main tests ---
    @patch("main.parse_file")
    @patch("main.get_embeddings")
    @patch("builtins.input", return_value="my ingredient")
    @patch("ollama.embeddings")
    @patch("main.find_most_similar")
    @patch("ollama.chat")
    @patch("main.alive_bar") # Mock context manager
    def test_main_function(self, mock_bar, mock_chat, mock_find, mock_ollama_emb, mock_input, mock_get_emb, mock_parse):
        # Setup mocks
        mock_parse.return_value = ["ing1", "ing2", "ing3"]
        mock_get_emb.return_value = [[0.1], [0.2], [0.3]]
        mock_ollama_emb.return_value = {"embedding": [0.5]}
        # Return index 0 as most similar
        mock_find.return_value = [(0.99, 0)]
        mock_chat.return_value = {"message": {"content": "Detected allergens"}}
        
        # Mock alive_bar to yield a dummy function
        mock_bar.return_value.__enter__.return_value = lambda: None

        main.main()

        # Verifications
        mock_parse.assert_called_with("data.json")
        mock_get_emb.assert_called()
        mock_input.assert_called()
        mock_ollama_emb.assert_called_with(model="embeddinggemma", prompt="my ingredient")
        mock_find.assert_called()
        mock_chat.assert_called()

if __name__ == "__main__":
    unittest.main()
