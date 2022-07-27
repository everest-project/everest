import {makeStyles} from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import React from 'react';
import TextField from '@material-ui/core/TextField';
import { Player } from 'video-react';
import "video-react/dist/video-react.css";
import {Fab} from "@material-ui/core";
import LocationSearchingIcon from '@material-ui/icons/LocationSearching';
import CardMedia from '@material-ui/core/CardMedia';
import Grid from "@material-ui/core/Grid";
import {ErrorHandler} from "./Util";
import {trackPromise} from "react-promise-tracker";


const useStyles = makeStyles((theme) => ({
    root: {
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent:"center",
        padding: "0em 0.5em"
    },
    video_preview: {
        width: "auto",
        flexGrow: 1,
        display: "block",
        padding: "0.5em 1.5em",
        overflow: "auto",
        height:"auto"
    },
    udf_preview: {
        width: "auto",
        display: "flex",
        flexDirection: "row",
        flexWrap: "wrap",
        padding: "0.5em 1.5em",
        overflow: "auto"
    },
    preview_title: {
        color: "#3f51b5"
    },
    video_box: {
        backgroundColor: "gray",
        flexGrow: 1,
        maxWidth: "100%",
        marginTop: "1em",
        marginBottom: "1em"
    }, 
    udf_visualize_box: {
        width: "100%"
    },
    udf_pic_box: {
        backgroundColor: "gray",
        width: "100%",
        height: 300,
        marginTop: "1em",
        marginBottom: "1em",
        flexGrow: 1,
    },
    udf_score_box: {
        flexGrow: 1,
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        width: "100%",
        flexWrap:"wrap"
    },
    score_text: {
        width: "60%"
    }
}));

function PreviewPanel(props) {
    const classes = useStyles()
    const video_name = props.query.video
    const video_base_path = process.env.PUBLIC_URL + "/resized_video"
    const video_url = video_base_path + "/" + video_name

    const [udf_image, set_udf_image] = React.useState("")
    const [score, set_score] = React.useState(0)

    let video_player;

    const handle_preview_icon_click = (event) => {
        event.preventDefault()
        let current_time = video_player.getState().player.currentTime
        const udf_name = props.query.udf
        const video_name = props.query.video
        let fps = props.video_param.fps
        let frame = Math.round(current_time * fps)

        trackPromise(fetch(`/api/test_udf/${udf_name}/${video_name}/${frame}`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body:JSON.stringify(props.query.udf_params)
            }).then(
            (ret) => {console.log(ret); return ret.json()}
        ).then((ret) => {
            set_score(Math.floor(ret.score * 100) / 100.0)
            console.log(ret.img)
            set_udf_image(ret.img)
        }).catch(ErrorHandler)
        )

        // trackPromise(fetch(`/api/test_udf/${udf_name}/${video_name}/${frame}`,
        //     {
        //         method: 'POST',
        //         headers: {
        //             'Content-Type': 'application/json'
        //         },
        //         body:JSON.stringify(props.query.udf_params)
        //     }).catch(ErrorHandler)
        // )
    }


    return (
        <Box className={classes.root}>
            <Box className={classes.video_preview}>
                <Typography variant="h5" className={classes.preview_title}>Video preview</Typography>
                <Player
                className={classes.video_box}
                src={video_url}
                ref={(ref) => video_player = ref}
                />
                {/*<Box className={classes.video_box}></Box>*/}
            </Box>
            <Box className={classes.video_preview} style={{textAlign:"center"}}>
                <Fab variant="extended" color="primary" size="medium"  onClick={handle_preview_icon_click}>
                        <LocationSearchingIcon />
                  &nbsp;UDF score preview
                </Fab>
            </Box>
            <Box className={classes.udf_preview}>
                <Typography variant="h5" className={classes.preview_title}>Score preview</Typography>
                <Box className={classes.udf_visualize_box}>
                    <CardMedia className={classes.udf_pic_box} image={udf_image}>
                    </CardMedia>
                    <Box className={classes.udf_score_box}>
                        <Box>
                            <Grid container spacing={1} alignItems="center">
                              <Grid item>
                                Score:
                              </Grid>
                              <Grid item>
                                <TextField value={score} />
                              </Grid>
                            </Grid>
                          </Box>
                    </Box>
                </Box>
            </Box>
        </Box>
    )
}

export default PreviewPanel;
