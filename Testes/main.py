import customtkinter as ctk
from customtkinter import filedialog as fd
import face_tracking_vegas_integration as ft
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.geometry("500x350")
frame = ctk.CTkFrame(master=root)
label = ctk.CTkLabel(master=frame, text='FTracking')
progress_bar = ctk.CTkProgressBar(master=frame)


def update_progress(current, total):
    progress_value = (current / total) * 100
    progress_bar.set(progress_value)
    root.update_idletasks()

def upload_file(): 
    try:
        filetypes = (('MP4 files', '*.mp4'), ('All files', '*.*')) 
        file = fd.askopenfile(filetypes=filetypes, initialdir="D:/Downloads")
        
        if file is None:  # Handle cancel action
            return
        
        input_path = file.name
        
        if not os.path.isfile(input_path):
            label.configure(text="Error: File not found.")
            return
        
        label.configure(text="Processing video, please wait...")
        progress_bar.set(0)  # Reset progress bar
        root.update_idletasks()

        # You might want to make output file and fps configurable
        output_path = f'{input_path}_coordinates.txt'
        ft.process_video(input_path, output_path, update_progress)
        
        label.configure(text="Processing completed!")
    except Exception as e:
        label.configure(text=f"Error: {str(e)}")

def main():
    frame.pack(pady=20, padx=60, fill='both', expand=True)

    label.pack(pady=12, padx=10)

    button = ctk.CTkButton(master=frame, text='Upload file', command=upload_file)
    button.pack(pady=12, padx=10)

    progress_bar.pack(pady=12, padx=10)
    progress_bar.set(0)

    root.mainloop()

if __name__ == '__main__':
    main()
