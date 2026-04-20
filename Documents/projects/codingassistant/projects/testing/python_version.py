# Check the latest stable Python version
import subprocess

def get_latest_python_version():
    """
    Get the latest stable Python version from the official Python website.
    
    Returns:
        str: The latest stable Python version.
    """
    try:
        result = subprocess.run(['curl', '-s', 'https://www.python.org/downloads/'], capture_output=True, text=True)
        output = result.stdout
        # Extract the latest version from the HTML content
        import re
        match = re.search(r'Latest Python 3 Release - Python (\d+\.\d+(\.\d+)?)', output)
        if match:
            return match.group(1)
        else:
            raise ValueError("Could not find the latest Python version.")
    except Exception as e:
        print(f"Error fetching the latest Python version: {e}")
        return None

if __name__ == "__main__":
    latest_version = get_latest_python_version()
    if latest_version:
        print(f"The latest stable Python version is: {latest_version}")