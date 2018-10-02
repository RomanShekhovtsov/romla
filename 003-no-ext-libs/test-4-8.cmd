@ECHO OFF
set MODE=classification
set FOLDER=..\..\check_4_c
call test.cmd %MODE% %FOLDER% %1 %2
IF NOT %ERRORLEVEL%==0 GOTO EXIT
set FOLDER=..\..\check_5_c
call test.cmd %MODE% %FOLDER% %1 %2
IF NOT %ERRORLEVEL%==0 GOTO EXIT
set FOLDER=..\..\check_6_c
call test.cmd %MODE% %FOLDER% %1 %2
IF NOT %ERRORLEVEL%==0 GOTO EXIT
set FOLDER=..\..\check_7_c
call test.cmd %MODE% %FOLDER% %1 %2
IF NOT %ERRORLEVEL%==0 GOTO EXIT
set FOLDER=..\..\check_8_c
call test.cmd %MODE% %FOLDER% %1 %2
IF NOT %ERRORLEVEL%==0 GOTO EXIT
:EXIT
EXIT /B %ERRORLEVEL%