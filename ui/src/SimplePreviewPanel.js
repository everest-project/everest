import {makeStyles} from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import Divider from '@material-ui/core/Divider';
import React from 'react';
import TextField from '@material-ui/core/TextField';
import { Player } from 'video-react';
import "video-react/dist/video-react.css";

const useStyles = makeStyles((theme) => ({
    root: {
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "row",
    },
    video_preview: {
        width: "50%",
        flexGrow: 1,
        display: "block",
        padding: "0.5em 1.5em",
        overflow: "auto",
        height:"auto"
    },
    udf_preview: {
        width: "50%",
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

function SimplePreviewPanel(props) {
    const classes = useStyles()

    const video_name = props.query.video
    const video_base_path = process.env.PUBLIC_URL + "/resized_video"
    const video_url = video_base_path + "/" + video_name

    return (
        <Box width="100%">
            <Typography variant="h6" className={classes.preview_title}>Video preview</Typography>
            <Player
            className={classes.video_box}
            src={video_url}
            />
        </Box>
    )
}

export default SimplePreviewPanel;