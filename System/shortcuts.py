import os
import subprocess
import sys

def create_shortcut(target_exe, target_args, icon_path, working_dir, shortcut_name):
    """Creates a Windows Shortcut using VBScript, automatically resolving the Desktop path."""
    vbs_path = os.path.join(os.environ["TEMP"], "create_lnk_pc.vbs")
    
    # We use SpecialFolders("Desktop") to handle OneDrive or redirected Desktop folders correctly.
    vbs_content = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sDesktop = oWS.SpecialFolders("Desktop")
sLinkFile = sDesktop & "\\{shortcut_name}.lnk"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{target_exe}"
oLink.Arguments = "{target_args.replace('"', '""')}"
oLink.IconLocation = "{icon_path}"
oLink.WorkingDirectory = "{working_dir}"
oLink.Save
"""
    try:
        with open(vbs_path, "w", encoding='utf-8') as f:
            f.write(vbs_content)
        
        subprocess.call(["cscript.exe", "/nologo", vbs_path])
        print(f"  > [V] Desktop Shortcut Created: {shortcut_name}")
    except Exception as e:
        print(f"  > [!] Shortcut Failed: {e}")
    finally:
        if os.path.exists(vbs_path):
            os.remove(vbs_path)

def setup_shortcuts():
    print("\n--- Initializing SerenityPC Desktop Link ---")
    system_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(system_dir)
    
    python_exe = sys.executable
    import re
    pythonw_exe = re.sub(r'python\.exe$', 'pythonw.exe', python_exe, flags=re.IGNORECASE) if "python.exe" in python_exe.lower() else python_exe
    main_py_path = os.path.join(base_dir, "main.py")
    
    icon_path = os.path.join(system_dir, "serenity.ico")
    if not os.path.exists(icon_path):
        print(f"  > [!] Warning: Icon not found at {icon_path}")

    create_shortcut(
        target_exe=pythonw_exe,
        target_args=f'"{main_py_path}"',
        icon_path=icon_path,
        working_dir=base_dir,
        shortcut_name="SerenityPC"
    )

if __name__ == "__main__":
    setup_shortcuts()
