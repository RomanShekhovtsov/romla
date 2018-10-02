@ECHO OFF
set MODE=regression
set FOLDER=..\..\check_1_r
call test.cmd %MODE% %FOLDER% %1 %2 
IF NOT %ERRORLEVEL%==0 GOTO EXIT
set FOLDER=..\..\check_2_r
call test.cmd %MODE% %FOLDER% %1 %2
IF NOT %ERRORLEVEL%==0 GOTO EXIT
set FOLDER=..\..\check_3_r
call test.cmd %MODE% %FOLDER% %1 %2
IF NOT %ERRORLEVEL%==0 GOTO EXIT
:EXIT
EXIT /B %ERRORLEVEL%
