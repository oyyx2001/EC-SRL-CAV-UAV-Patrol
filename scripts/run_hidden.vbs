' Launch one command without a visible console and return immediately.
If WScript.Arguments.Count <> 1 Then
  WScript.Quit 2
End If
CreateObject("WScript.Shell").Run WScript.Arguments.Item(0), 0, False
