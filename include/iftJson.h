/**
 * @file iftJson.h
 * @brief Json parser and manipulation.
 * @author Samuel Martins
 * @date Oct 8, 2015
 * @ingroup Json
 * 
 * @note An example of Json reading and writing and how to get elements from it can be found in
 * ift/demo/Miscellaneous/DataStrutuctures/iftTestJson1.c
 * @note An example of how to create a Json and add elements to it can be found in
 * ift/demo/Miscellaneous/DataStrutuctures/iftTestJson2.c
 * 
 * This is an adaption and extension from the cJSON parser:
 * https://github.com/kbranigan/cJSON
 */
/*
  Copyright (c) 2009 Dave Gamble

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#ifndef _IFT_JSON_H_
#define _IFT_JSON_H_

#ifdef __cplusplus
extern "C" {
#endif


#include "iftCommon.h"
#include "iftDict.h"
#include "iftFile.h"
#include "iftMatrix.h"


/**
 * @brief Reads a Json file from the disk and returns a dictionary
 * @author Samuel Martins
 * @date Sep 17, 2017
 *
 * @param json_pathname The pathname from the Json to be read.
 * @return Resulting dictionary.
 */
iftDict *iftReadJson(const char *json_path, ...);


/**
 * @brief Writes a dict to a JSON file.
 * @author Samuka Martins
 * @date Sep 17, 2017
 */
void iftWriteJson(const iftDict *dict, const char *json_path, ...);


/**
 * @brief Writes a dict to a JSON file in a minified way.
 * @author Samuka Martins
 * @date Sep 17, 2017
 */
void iftWriteJsonMinified(const iftDict *dict, const char *json_path, ...);



#ifdef __cplusplus
}
#endif

#endif //_IFT_JSON_H_
