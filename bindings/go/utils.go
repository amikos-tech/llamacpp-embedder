package llama_embedder

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"strings"
)

func expandTilde(path string) (string, error) {
	if strings.HasPrefix(path, "~/") {
		usr, err := user.Current()
		if err != nil {
			return "", err
		}
		return filepath.Join(usr.HomeDir, path[2:]), nil
	}
	return path, nil
}

func extractTarGz(tarGzPath, destPath string) error {
	f, err := os.Open(tarGzPath)
	if err != nil {
		return fmt.Errorf("could not open tar.gz file: %v", err)
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("could not create gzip reader: %v", err)
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("tar reading error: %v", err)
		}

		target := filepath.Join(destPath, header.Name)

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return fmt.Errorf("could not create directory: %v", err)
			}
		case tar.TypeReg:
			f, err := os.Create(target)
			if err != nil {
				return fmt.Errorf("could not create file: %v", err)
			}
			if _, err := io.Copy(f, tr); err != nil {
				f.Close()
				return fmt.Errorf("could not copy file contents: %v", err)
			}
			f.Close()
		}
	}
	return nil
}

func extractZip(zipPath, destPath string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return fmt.Errorf("could not open zip file: %v", err)
	}
	defer r.Close()

	for _, f := range r.File {
		target := filepath.Join(destPath, f.Name)

		if f.FileInfo().IsDir() {
			err = os.MkdirAll(target, 0755)
			if err != nil {
				return fmt.Errorf("could not create directory: %v", err)
			}
			continue
		}

		if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return fmt.Errorf("could not create directory: %v", err)
		}

		outFile, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return fmt.Errorf("could not open output file: %v", err)
		}

		rc, err := f.Open()
		if err != nil {
			outFile.Close()
			return fmt.Errorf("could not open file in archive: %v", err)
		}

		_, err = io.Copy(outFile, rc)
		outFile.Close()
		rc.Close()

		if err != nil {
			return fmt.Errorf("could not copy file contents: %v", err)
		}
	}
	return nil
}

func getOSSharedLibName() string {
	switch cos := strings.ToLower(runtime.GOOS); cos {
	case "darwin":
		return "libllama-embedder.dylib"
	case "windows":
		return "llama-embedder.dll"
	default:
		return "libllama-embedder.so"
	}
}

func ensureCacheDir() error {
	if _, err := os.Stat(defaultCacheDir); os.IsNotExist(err) {
		err := os.MkdirAll(defaultCacheDir, 0755)
		if err != nil {
			return fmt.Errorf("could not create cache directory: %v", err)
		}
		err = os.MkdirAll(defaultModelCacheDir, 0755)
		if err != nil {
			return fmt.Errorf("could not create model cache directory: %v", err)
		}
		err = os.MkdirAll(defaultLibCacheDir, 0755)
		if err != nil {
			return fmt.Errorf("could not create library cache directory: %v", err)
		}
	}

	return nil
}
