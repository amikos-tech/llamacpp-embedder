package llama_embedder

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// ensureLibrary ensures that the shared library is downloaded and extracted. If it already exists, it will not be downloaded again.
// It returns the path to the shared library file.
func ensureLibrary(libraryVersion string) (string, error) {
	//llama-embedder-macos-arm64-v0.0.7.tar.gz
	var cos string
	var carch string
	var libArchiveExt string
	if runtime.GOOS == "darwin" {
		cos = "macos"
		libArchiveExt = "tar.gz"
		if runtime.GOARCH == "arm64" {
			carch = "arm64"
		} else {
			carch = "x64"
		}
	} else if runtime.GOOS == "linux" {
		cos = "linux"
		libArchiveExt = "tar.gz"
		if runtime.GOARCH == "arm64" {
			carch = "arm64"
		} else {
			carch = "x64"
		}
	} else if runtime.GOOS == "windows" {
		cos = "win"
		carch = "x64"
		libArchiveExt = "zip"
	} else {
		fmt.Println("Unsupported OS")
		os.Exit(1)
	}
	// https://github.com/amikos-tech/llamacpp-embedder/releases/download/go%2F<VERSION>/llama-embedder-<COS>-<CARCH>-<VERSION>.<EXT>
	var libArchiveBase = "llama-embedder-" + cos + "-" + carch + "-" + libraryVersion
	var sharedLibFilePath = filepath.Join(defaultLibCacheDir, libArchiveBase, getOSSharedLibName())
	if _, err := os.Stat(filepath.Join(defaultLibCacheDir, libArchiveBase)); err == nil {
		return sharedLibFilePath, nil
	}
	url := "https://github.com/amikos-tech/llamacpp-embedder/releases/download/go%2F" + libraryVersion + "/" + libArchiveBase + "." + libArchiveExt
	fmt.Printf("Downloading library from %s\n", url)
	segments := strings.Split(url, "/")
	filename := segments[len(segments)-1]
	if filename == "" {
		return "", fmt.Errorf("failed to extract filename from URL")
	}

	// Create the output file
	libArchive := filepath.Join(defaultLibCacheDir, filename)
	err := downloadFile(libArchive, url)
	if err != nil {
		return "", err
	}

	if cos == "win" {
		// Unzip the file
		err = extractZip(libArchive, filepath.Join(defaultLibCacheDir, libArchiveBase))
		if err != nil {
			return "", err
		}
	} else {
		// Untar the file
		err = extractTarGz(libArchive, filepath.Join(defaultLibCacheDir, libArchiveBase))
		if err != nil {
			return "", err
		}
	}

	return sharedLibFilePath, nil

}

// downloadFile downloads a file from a URL and saves it to the specified filepath.
func downloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to make HTTP request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer func(out *os.File) {
		err := out.Close()
		if err != nil {
			fmt.Println("failed to close file:", err)
		}
	}(out)

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to copy file contents: %w", err)
	}

	return nil
}

// downloadHFModel downloads a model from Hugging Face and saves it to the specified target location.
func downloadHFModel(hfRepo, hfFile, targetLocation, hfToken string) error {
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
