import {makeStyles} from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import Divider from '@material-ui/core/Divider';
import StepIcon from '@material-ui/core/StepIcon';
import MenuItem from '@material-ui/core/MenuItem';
import Select from '@material-ui/core/Select';
import OutlinedInput from '@material-ui/core/OutlinedInput';
import Fab from '@material-ui/core/Fab';
import CloudUploadIcon from '@material-ui/icons/CloudUpload';
import Grid from '@material-ui/core/Grid';
import React, {useEffect} from 'react';
import TextField from '@material-ui/core/TextField';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogTitle from '@material-ui/core/DialogTitle';
import Editor from "react-simple-code-editor";
import { highlight, languages } from "prismjs/components/prism-core";
import "prismjs/components/prism-python";
import "prismjs/themes/prism.css";
import { trackPromise } from 'react-promise-tracker';
import PreviewPanel from "./PreviewPanel";
import SimplePreviewPanel from "./SimplePreviewPanel";
import InsertDriveFileIcon from '@material-ui/icons/InsertDriveFile';

const useStyles = makeStyles((theme) => ({
  root: {
    width: "100%",
    height: "100%",
  },
  stepbox: {
    margin: "1em",
    paddingBottom: "0.5em"
  },
  step_title: {
    display: "flex",
    flexDirection: "row",
    alignItems: "center",
    marginBottom: "0.5em"
  },
  step_title_text: {
    marginLeft: "1em",
    color: "#3f51b5"
  },
  step_content: {
    marginLeft: "3em",
    color: "#54617a"
  },
  step_row: {
    display: "flex",
    flexDirection: "row",
    alignItems: "center"
  },
  video_select: {
    minWidth: "60%",
  },
  upload_button: {
    marginLeft: "2em",
    width:"15em"
  },
  video_param: {
    marginTop: "1em",
    width: "25em"
  },
  video_param_item: {
    display: "flex",
    flexDirection: "row",
    alignItems: "flex-end"
  },
  video_param_key: {
    marginRight: "1em"
  },
  edit_button: {
    marginLeft: "0.5em",
  },
  run_box: {
    marginTop: "5em",
    display: "flex",
    justifyContent: "center"
  },
  run_button: {
    width: "10em",
    margin:"0em 1em"
  },
  udf_editor_udf_name_textfield:{
    display:"inline"
  },
  querypane:{
    display:"flex",

  },
  select_box:{
    margin:"0em 0.3em"
  }
}));

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// DOM of upload input box
let ref_input_upload = {};

function SimpleSelectVideo(props) {
  const classes = useStyles();
    // state of video list
  const [video_list, set_video_lists] = React.useState([])

  const get_video_list = () => {
    let url = "/api/video_list";
    fetch(url).then(
        res => res.json()
    ).then(
        (result) => {
          set_video_lists(result)
        },
        (error) => {
          console.log("Fail to fetch")
        }
    );
  }

  const on_select_video = (event) => {
    let video_name = video_list[event.target.value]

    fetch(`/api/video_params/${video_name}`).then(
        res => res.json()
    ).then(
        (ret) => {
          console.log("video param", ret)
          props.on_video_change(ret)
        },
        (err) =>{
          console.log("Fail to fetch video param", err)
        }
    )
    props.on_query_change({video: video_name})
  }

  const upload = () => {
    console.log("Trigger upload")
    var data = new FormData();
    var videodata = ref_input_upload.files[0];
    data.append("file", videodata);

    trackPromise(fetch("/api/video", {
      method: "POST",
      body: data
    }).then(function (res) {
      get_video_list()
    }, function (e) {
      alert("Error submitting form!");
    }));
  }

  // todo: upload reload list
  useEffect(() => {
      console.log("Debug useEffect get_video_list()")

    // Update the document title using the browser API
    get_video_list();
  }, [1]);

  return (
    <Box className={classes.step_content}>
      <Box className={classes.step_row}>
        <Select id="video_select" className={classes.video_select} onChange={on_select_video} input={<OutlinedInput margin="dense" />}>
          {video_list.map((elm, idx) => (<MenuItem value={idx}>{elm}</MenuItem>))}
        </Select>

        <input id="input_upload_video" type="file" ref={(ref) => ref_input_upload = ref} style={{ display: 'none' }} onChange={upload}/>
        <Fab color="primary" variant="extended" className={classes.upload_button} onClick={(e) => ref_input_upload.click()}>
          <CloudUploadIcon />
              &nbsp;&nbsp;Upload
        </Fab>
      </Box>
    </Box>
  )
}

const _get_udf_list = (result_handler) => {
    let url = "/api/udf_list";
    fetch(url).then(
        res => res.json()
    ).then(
        (result) => {
          result_handler(result)
        },
        (error) => {
          console.log("Fail to fetch udf_list", error)
        }
    );
  }

const _get_udf_param = (udf_name, udf_name_param_handler) => {
   let url = `/api/udf_params/${udf_name}/params`;
    fetch(url).then(
        res => res.json()
    ).then(
        (result) => {
          udf_name_param_handler(result, udf_name)
        },
        (error) => {
          console.log("Fail to fetch udf params", error)
        }
    );
}

const k_list = [1,5,10,20,50]

const window_list = ["1 frame", "1s", "10s", "30s", "60s"]
const window_display_name = ["frame", "1s-clips", "10s-clips", "30s-clips", "60s-clips"]


function SimpleSelectQueryPane(props) {
  const k = props.query.k
  const udf = props.query.udf
  const window = props.query.window
  const on_query_change = props.on_query_change
  const udf_list = props.udf_list
  const set_udf_list = props.set_udf_list

  const classes = useStyles()

  const create_update_udf_params = (udf_params, udf_name) => {
    on_query_change({"udf": udf_name, "udf_params": udf_params})
  }

  const udf_select_handler = (udf_name) => {
    _get_udf_param(udf_name, create_update_udf_params)
  }

  const get_udf_params = (udf_name) => {
    console.log("fetch udf param with name ",udf_name)
    _get_udf_param(udf_name, create_update_udf_params)
  }

  /*
  useEffect(() => _get_udf_list((ret)=>{
      console.log("Debug use effect _get_udf_list()")
      set_udf_list(ret)
      if(udf === ""){
        get_udf_params(ret[0])
      }
    }), [1])
    */

  return (
      <div className={classes.querypane}>
          <Typography variant="h6">Top - </Typography>
          <Select className={classes.select_box} autoWidth={true} input={<OutlinedInput margin="dense" />} value={k} onChange={(event)=>on_query_change({"k":event.target.value})}>
            {k_list.map(elm => (<MenuItem value={elm}>{elm}</MenuItem>))}
          </Select>
          <Select className={classes.select_box} autoWidth={true} input={<OutlinedInput margin="dense" />} value={udf} onChange={(event)=>_get_udf_param(event.target.value, create_update_udf_params)}>
            {udf_list.map(elm => (<MenuItem value={elm}>{elm}</MenuItem>))}
          </Select>
          <Select className={classes.select_box} autoWidth={true} input={<OutlinedInput margin="dense" />} value={window} onChange={(event)=>on_query_change({"window":event.target.value})}>
            {window_list.map((elm, idx) => (<MenuItem value={elm}>{window_display_name[idx]}</MenuItem>))}
          </Select>
      </div>
  )
}

function UDFEditor(props) {
  const set_udf_list = props.set_udf_list
  const udf_name = props.query.udf
  const on_query_change = props.on_query_change
  const udf_params = props.query.udf_params

  const create_update_udf_params = (udf_params, udf_name) => {
    on_query_change({"udf": udf_name, "udf_params": udf_params})
  }

  const on_enter_udf_param = (event) => {
    var key = event.target.id.slice(3)
    let new_udf_params = udf_params
    new_udf_params[key] = event.target.value

    create_update_udf_params(new_udf_params, udf_name)
  }

  const handle_udf_editor_udf_name_change = (event) => {
    let udf_name=event.target.value
    on_query_change({"udf": udf_name, "udf_params": {}})
  }

  const classes = useStyles()

  const [code_state, set_code] = React.useState("")

  const pull_code_state = (udf_name) => {
    console.log("code", udf_name)
    if (udf_name == "") {
      set_code("")
    } else {
      trackPromise(
        fetch(`/api/udf_list/${udf_name}`).then(
            (ret) => ret.json()
        ).then((ret) => {
          set_code(ret.udf_content)
        })
      )
    }
  };

  useEffect(()=>{
    console.log("Debug use effect pull_code_state()")
    pull_code_state(udf_name)}, [1])



  const handle_udf_editor_save = () => {
    /*
    trackPromise(
        fetch(`/api/udf_list/${udf_name}`,
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({"udf_content": code_state})
            }).then(()=>{
                  _get_udf_list(set_udf_list)
        })
    )
    */
  }

  return (
      <Box style={{color: "#54617a", width:"auto"}}>
        <Grid container spacing={3} className={classes.video_param}>
          <Grid item xs={10} className={classes.video_param_item}>
            <Typography variant="subtitle1" className={classes.video_param_key}>UDF name:</Typography>
            <TextField value={udf_name} onChange={handle_udf_editor_udf_name_change}/>
          </Grid>

        {/*comment out UDF param*/}
        {/*{*/}
        {/*  Object.keys(udf_params).map((key, idx) => (*/}
        {/*    <Grid item xs={6} className={classes.video_param_item}>*/}
        {/*      <Typography variant="subtitle1" className={classes.video_param_key}>{key}:</Typography>*/}
        {/*      <TextField value={udf_params[key]} id={"udf" + key} onChange={on_enter_udf_param}/>*/}
        {/*    </Grid>*/}
        {/*  ))*/}
        {/*}*/}

        </Grid>
        <Editor
          value={code_state}
          onValueChange={(code) => set_code(code)}
          highlight={(code) => highlight(code, languages.py)}
          padding={10}
          style={{
            fontFamily: '"Fira code", "Fira Mono", monospace',
            fontSize: 12,
            height:500,
            border:"solid 1px",
            overflow: "auto",
            marginTop:"5em"
          }}
        />
        <Box style={{"text-align":"center"}}>
          <Fab style={{"margin":"1em"}} variant="extended" onClick={handle_udf_editor_save} color="primary">
            Save
          </Fab>
        </Box>
      </Box>
  )
}

function convert_udf_window(in_window, fps){
  let split_in_window = in_window.split(" ")
  if(split_in_window.length === 1){
    //second
    let num_second = parseInt(split_in_window[0].slice(0, -1))
    return Math.round(num_second * fps)
  }else if (split_in_window.length === 2){
    //frame
    return parseInt(split_in_window[0])
  }else{
    throw `Unknown window: ${in_window}`
  }

}
function SimpleQueryPanel(props) {
  const video_param = props.video_param
  const query = props.query
  const set_result_img_paths = props.set_result_img_paths
  const on_query_change = props.on_query_change

  const classes = useStyles();

  const [is_advance_open, set_is_advance_open] = React.useState(false)
  const [udf_list, set_udf_list] = React.useState(["number_of_cars", "happy_moment","exciting_moment"])
/*  const init_udf =()=> {
    _get_udf_list( (result)=>{
	set_udf_list(result)
      })
  }
  
  init_udf()
*/
  const handle_udf_editor_close = () => {
    set_is_advance_open(false)
    set_udf_list(["number_of_cars", "happiest", "tailgating_danger"])
  }

  const pull_result = (job_id) => {
    trackPromise(fetch(`/api/run/${job_id}`).then(
          (ret) => ret.json()).then(
          (ret)=> {
            let ready = ret.ready
            if(ready){
              let isSuccess = ret.success
              let ret_objs = ret.results
              console.log("Successfully pull result", ret_objs)
              set_result_img_paths(ret_objs)
            }else{
              let timeout = 5
              console.log(`${job_id} is not ready, pull after ${timeout} second`)
              return sleep(5*1000).then(()=>pull_result(job_id))
            }
          }
    ))
  }

  const on_click_run = (event) => {

    let query = {...props.query}
    console.log("org window: ",query["window"])

    query["window"] = convert_udf_window(query["window"], video_param.fps)
    console.log("Converted window: ",query["window"])
    trackPromise(
      fetch("/api/run", {
                  method: 'POST',
                  headers: {
                  'Content-Type': 'application/json'
                  },
                  body:JSON.stringify(query)
              }).then((ret) => ret.json()).then(
          (ret) => {
            let job_id = ret.job_id
            console.log("Get job_id: ", job_id)
            pull_result(job_id)
          })
    )
  }

  const create_update_udf_params = (udf_params, udf_name) => {
    on_query_change({"udf": udf_name, "udf_params": udf_params})
  }

  const handle_new_udf_click = () => {
    create_update_udf_params({}, "")
  }

  return (
    <div className={classes.root}>
      <Box className={classes.stepbox}>
        <Box className={classes.step_title}>
          <StepIcon active={true} icon={1} />
          {/*Remove underline of the info*/}
          <Typography variant="h5" className={classes.step_title_text}>Select a Video</Typography>
        </Box>
        <SimpleSelectVideo on_query_change={props.on_query_change} on_video_change={props.on_video_change}/>
        <Box className={classes.step_content}>
          <Box className={classes.step_row}>
            <SimplePreviewPanel query={query} video_param={video_param}/>
          </Box>
        </Box>
      </Box>
      <Divider />
      <Box className={classes.stepbox}>
        <Box className={classes.step_title}>
          <StepIcon active={true} icon={2} />
          <Typography variant="h5" className={classes.step_title_text}>Query</Typography>
        </Box>
        <Box className={classes.step_content}>
        <Fab color="secondary" size="large" variant="extended" className={classes.run_button} disabled={props.query.video === undefined || props.query.udf === ""} onClick={on_click_run}>
          RUN
        </Fab>
   <SimpleSelectQueryPane on_query_change={props.on_query_change} query={props.query}
          set_udf_list={set_udf_list} udf_list={udf_list}/>
        </Box>
      </Box>
      <Box className={classes.run_box}>
        <Fab color="primary" size="large" variant="extended" className={classes.run_button} disabled={props.query.video === undefined || props.query.udf === "" } onClick={()=>set_is_advance_open(true)} >
          Advanced...
        </Fab>
        <Fab color="secondary" size="large" variant="extended" className={classes.run_button} disabled={props.query.video === undefined || props.query.udf === ""} onClick={on_click_run}>
          RUN
        </Fab>
      </Box>

      {/*Advance box*/}
      <Dialog style={{zIndex:1200}} open={is_advance_open} onClose={()=>set_is_advance_open(true)} aria-labelledby="form-dialog-title" fullWidth={true} maxWidth="xl">
          <DialogTitle id="form-dialog-title">UDF Editor</DialogTitle>

          <DialogContent>
            {/*2 columns*/}
            <Box display="flex" flexDirection="row">
              <Box width="50%">
                {/*row base*/}
                <UDFEditor set_udf_list={set_udf_list} query={props.query}
                  on_query_change={props.on_query_change}
                />
              </Box>
              <Box width="50%">
                <PreviewPanel query={props.query} video_param={props.video_param}/>
              </Box>
            </Box>
          </DialogContent>
          <DialogActions>
            <Fab style={{"width":"10em"}} variant="extended" color={"primary"} onClick={handle_new_udf_click}>
              <InsertDriveFileIcon/>
              New UDF
            </Fab>
            <Fab onClick={handle_udf_editor_close} variant="extended" color="primary">
              Close
            </Fab>
          </DialogActions>
        </Dialog>
    </div>
  );
}

export default SimpleQueryPanel;
