package utils

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// downloadHFModel downloads a model from Hugging Face and saves it to the specified target location.
func DownloadHFModel(hfRepo, hfFile, targetLocation, hfToken string) error {
	if hfFile == "" || hfRepo == "" {
		return fmt.Errorf("hfRepo and hfFile are required")
	}
	if !strings.HasSuffix(strings.ToLower(hfFile), ".gguf") {
		return fmt.Errorf("model file must be a .gguf file")
	}

	// Validate and sanitize the filename
	filename := sanitizeFileName(filepath.Base(hfFile))
	if filename == "" {
		return fmt.Errorf("invalid filename")
	}

	// Construct and validate the URL
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", sanitizeURLPath(hfRepo), sanitizeURLPath(hfFile))
	if !isValidHuggingFaceURL(url) {
		return fmt.Errorf("invalid Hugging Face URL")
	}

	// Determine the output path
	outputPath, err := determineOutputPath(targetLocation, filename)
	if err != nil {
		return err
	}

	// Check if the file already exists
	if _, err := os.Stat(outputPath); err == nil {
		return nil // File already exists, no need to download
	}

	// Create the HTTP client and request
	client := &http.Client{}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}

	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}

	// Execute the request
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP error: %s", resp.Status)
	}

	// Create the output file
	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	// Write response body to file
	_, err = io.Copy(outFile, resp.Body)
	return err
}

// EnsureModels ensures that the models are downloaded and available in the cache directory.
func EnsureModels(models string) error {
	modelList := strings.Split(models, ";")
	for _, model := range modelList {
		segments := strings.Split(model, "/")
		if len(segments) < 3 {
			return fmt.Errorf("invalid model format: %s", model)
		}
		hfRepo := fmt.Sprintf("%s/%s", segments[0], segments[1])
		hfFile := strings.Join(segments[2:], "/")
		if !strings.HasSuffix(strings.ToLower(hfFile), ".gguf") {
			return fmt.Errorf("model file must be a .gguf file")
		}
		// Validate the file name
		if !isValidFileName(hfFile) {
			return fmt.Errorf("invalid file name: %s", hfFile)
		}

		targetLocation := filepath.Join(GetModelCacheDir(), sanitizeFileName(hfFile))
		fmt.Printf("Downloading model %s from %s to %s\n", hfFile, hfRepo, targetLocation)
		err := DownloadHFModel(hfRepo, hfFile, targetLocation, "")
		if err != nil {
			return fmt.Errorf("Error downloading model %s: %v\n", hfFile, err)
		}
	}
	return nil
}

// isValidFileName checks if the given file name is valid and safe
func isValidFileName(name string) bool {
	// Check for any path traversal attempts
	if strings.Contains(name, "..") {
		return false
	}

	// Check for any directory separators
	if strings.ContainsAny(name, "/\\") {
		return false
	}

	// Additional checks can be added here if needed

	return true
}

// sanitizeFileName removes any potentially dangerous characters from the file name
func sanitizeFileName(name string) string {
	// Remove any characters that aren't alphanumeric, dash, underscore, or dot
	return regexp.MustCompile(`[^a-zA-Z0-9\-_.]`).ReplaceAllString(name, "")
}

// sanitizeURLPath sanitizes the path component of a URL
func sanitizeURLPath(path string) string {
	// Remove any characters that aren't alphanumeric, dash, underscore, dot, or forward slash
	return regexp.MustCompile(`[^a-zA-Z0-9\-_./]`).ReplaceAllString(path, "")
}

// isValidHuggingFaceURL checks if the given URL is a valid Hugging Face URL
func isValidHuggingFaceURL(url string) bool {
	return strings.HasPrefix(url, "https://huggingface.co/") && strings.Contains(url, "/resolve/main/")
}

// determineOutputPath determines the final output path for the downloaded file
func determineOutputPath(targetLocation, filename string) (string, error) {
	if targetLocation == "" {
		return filename, nil
	}

	info, err := os.Stat(targetLocation)
	if err == nil && info.IsDir() {
		return filepath.Join(targetLocation, filename), nil
	}

	if os.IsNotExist(err) && strings.HasSuffix(targetLocation, string(os.PathSeparator)) {
		if err := os.MkdirAll(targetLocation, os.ModePerm); err != nil {
			return "", fmt.Errorf("failed to create directory: %v", err)
		}
		return filepath.Join(targetLocation, filename), nil
	}

	// targetLocation is a file path
	outputDir := filepath.Dir(targetLocation)
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		return "", fmt.Errorf("failed to create directory: %v", err)
	}
	return targetLocation, nil
}
