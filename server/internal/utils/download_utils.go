package utils

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
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

	if _, err := os.Stat(targetLocation); err == nil {
		return nil
	}
	client := &http.Client{}

	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", hfRepo, hfFile)

	// Create HTTP GET request
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}

	// Set Authorization header if HF_TOKEN is provided
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}

	// Execute the request
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check for HTTP errors
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP error: %s", resp.Status)
	}

	// Extract filename from URL
	segments := strings.Split(url, "/")
	filename := segments[len(segments)-1]
	if filename == "" {
		return fmt.Errorf("failed to extract filename from URL")
	}

	var outputPath string

	// Determine the output path based on targetLocation
	if targetLocation != "" {
		// Check if targetLocation is a directory
		info, err := os.Stat(targetLocation)
		if err == nil && info.IsDir() {
			// targetLocation is an existing directory
			outputPath = filepath.Join(targetLocation, filename)
		} else if os.IsNotExist(err) && strings.HasSuffix(targetLocation, string(os.PathSeparator)) {
			// targetLocation is a non-existing directory (ends with / or \)
			if err := os.MkdirAll(targetLocation, os.ModePerm); err != nil {
				return fmt.Errorf("failed to create directory: %v", err)
			}
			outputPath = filepath.Join(targetLocation, filename)
		} else {
			// targetLocation is a file path
			outputDir := filepath.Dir(targetLocation)
			// Create the directory if it doesn't exist
			if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
				return fmt.Errorf("failed to create directory: %v", err)
			}
			outputPath = targetLocation
		}
	} else {
		// No targetLocation provided, use current directory
		outputPath = filename
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
		hfRepo := fmt.Sprintf("%s/%s", segments[0], segments[1])
		hfFile := strings.Join(segments[2:], "/")
		if !strings.HasSuffix(strings.ToLower(hfFile), ".gguf") {
			return fmt.Errorf("model file must be a .gguf file")
		}
		targetLocation := filepath.Join(GetModelCacheDir(), hfFile)
		fmt.Printf("Downloading model %s from %s to %s\n", hfFile, hfRepo, targetLocation)
		err := DownloadHFModel(hfRepo, hfFile, targetLocation, "")
		if err != nil {
			return fmt.Errorf("Error downloading model %s: %v\n", hfFile, err)
		}
	}
	return nil
}
