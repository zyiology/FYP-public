#Requires AutoHotkey v2.0

^!r::Reload

^!f:: {
	init := 3964 ; 1321, 857, 681
	Loop 1000 ; Adjust the 10 to however many times you want to loop 4451-1323
	{
		Send "^c" ; Ctrl + C
		Sleep 1000 ; Wait a bit for the action to complete
		Send "{Down}"
		Sleep 300
		Send "!{Tab}" ; Alt + Tab
		Sleep 1000
		Send "^v" ; Ctrl + V
		Sleep 500
		MouseClick "right" ; Right Click
		Sleep 500
		Loop 5
		{
			Send "{Up}" ; Arrow Down *9
			Sleep 100
		}
		Send "{Enter}"
		Sleep 500
		fileName := "image_" . init . ".png"
		Send fileName ; Type "image_[loop number].png"
		Sleep 500
		Send "{Enter}" ; Enter
		Sleep 300
		Send "{Delete}" ; Delete
		Sleep 300
		Send "!{Tab}" ; Alt Tab
		Sleep 1000
		init++
	}
}